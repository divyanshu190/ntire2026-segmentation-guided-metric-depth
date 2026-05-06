import argparse
import csv
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IndoorDataset
from loss import CombinedSegmentationLoss

warnings.filterwarnings("ignore", category=UserWarning)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LoRALinear(torch.nn.Module):
    def __init__(self, linear, rank=4, alpha=1.0):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / rank
        in_f, out_f  = linear.in_features, linear.out_features
        self.lora_A  = torch.nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B  = torch.nn.Parameter(torch.zeros(out_f, rank))
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        self.to(linear.weight.device)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling


def apply_lora(model, rank=4):
    if not hasattr(model, "sam_mask_decoder"):
        print("[WARNING] No sam_mask_decoder — skipping LoRA.")
        return 0
    count = 0
    for name, module in list(model.sam_mask_decoder.named_modules()):
        if isinstance(module, torch.nn.Linear) and module.in_features >= 64:
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model.sam_mask_decoder
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            setattr(parent, child_name, LoRALinear(module, rank=rank))
            count += 1
    return count


def build_model(ckpt_path, device):
    import sam2 as _sam2_pkg
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    config_dir = str(Path(_sam2_pkg.__file__).parent / "configs" / "sam2")
    print(f"Config dir: {config_dir}")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="sam2_hiera_l")

    print("Config loaded. Instantiating model...")
    model = instantiate(cfg.model)
    print(f"Model type: {type(model)}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(f"Checkpoint loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    model.to(device)
    return model


def forward_sam2(model, imgs):
    import torch.nn.functional as F
    B = imgs.shape[0]
    dec = model.sam_mask_decoder


    backbone_out   = model.image_encoder(imgs)
    img_embeddings = backbone_out["vision_features"]  
    fpn            = backbone_out["backbone_fpn"]





    feat_s1 = dec.conv_s1(fpn[1])   
    feat_s0 = dec.conv_s0(fpn[0])   
    high_res_features = [feat_s0, feat_s1]


    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
            points=None, boxes=None, masks=None
        )
        sparse_embeddings = sparse_embeddings.expand(B, -1, -1)
        dense_embeddings  = dense_embeddings.expand(B, -1, -1, -1)


    low_res_masks, _, _, _ = dec(
        image_embeddings=img_embeddings,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )

    return F.interpolate(low_res_masks, size=(imgs.shape[-2], imgs.shape[-1]),
                         mode="bilinear", align_corners=False)


@torch.no_grad()
def compute_iou(logits, targets):
    preds   = (torch.sigmoid(logits) > 0.5).float().view(logits.size(0), -1)
    targets = (targets > 0.5).float().view(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()


def train(args):
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = IndoorDataset(args.data, split="train")
    val_ds   = IndoorDataset(args.data, split="val")
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = build_model(args.ckpt, device)

    for p in model.image_encoder.parameters():
        p.requires_grad = False
    if hasattr(model, "memory_encoder"):
        for p in model.memory_encoder.parameters():
            p.requires_grad = False
    if hasattr(model, "sam_prompt_encoder"):
        for p in model.sam_prompt_encoder.parameters():
            p.requires_grad = False

    n_lora = apply_lora(model, rank=args.lora_rank)
    print(f"LoRA layers: {n_lora}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.wd)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1.0 + np.cos(np.pi * (epoch - warmup) / max(1, args.epochs - warmup)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = CombinedSegmentationLoss(w_bce=0.3, w_dice=0.4, w_focal=0.2, w_iou=0.1)
    scaler    = GradScaler("cuda", enabled=(device == "cuda"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_iou", "lr"])

    best_iou = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{args.epochs} [train]")
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            with autocast("cuda", enabled=(device == "cuda")):
                pred_masks = forward_sam2(model, imgs)
                loss, components = criterion(pred_masks, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{components['dice']:.4f}")

        avg_train = np.mean(train_losses)
        model.eval()
        val_losses, val_ious = [], []
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{args.epochs} [val]  "):
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast("cuda", enabled=(device == "cuda")):
                    pred_masks = forward_sam2(model, imgs)
                    loss, _ = criterion(pred_masks, masks)
                val_losses.append(loss.item())
                val_ious.append(compute_iou(pred_masks, masks))

        avg_val = np.mean(val_losses)
        avg_iou = np.mean(val_ious)
        cur_lr  = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:03d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | IoU: {avg_iou:.4f} | LR: {cur_lr:.2e}")

        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_iou": best_iou}, out_dir / "sam2_best.pth")
            print(f"  checkpoint saved (IoU={best_iou:.4f})")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch+1, avg_train, avg_val, avg_iou, cur_lr])

        scheduler.step()

    torch.save(model.state_dict(), out_dir / "sam2_final.pth")
    print(f"\nDone! Best IoU: {best_iou:.4f} | Saved to: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      default="train")
    p.add_argument("--ckpt",      default="sam2_hiera_large.pt")
    p.add_argument("--out",       default="models")
    p.add_argument("--epochs",    type=int,   default=60)
    p.add_argument("--batch",     type=int,   default=1)
    p.add_argument("--lr",        type=float, default=3e-4)
    p.add_argument("--wd",        type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
