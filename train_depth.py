

import argparse
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



def load_calibration(calib_path):
    focal, baseline = 3757.0, 0.12
    try:
        tree = ET.parse(calib_path)
        for elem in tree.getroot().iter():
            tag  = (elem.tag or "").lower()
            name = (elem.get("name") or "").lower()
            text = (elem.text or "").strip()
            if ("p1" in tag or "p1" in name) and text:
                focal = float(text.split()[0])
            if (tag == "t" or name == "t") and text:
                baseline = abs(float(text.split()[0])) / 1000.0
    except Exception:
        pass
    return focal, baseline


def disp_to_depth_cm(disp, focal, baseline):
    valid = disp > 0
    depth = np.zeros_like(disp, dtype=np.float32)
    depth[valid] = (focal * baseline) / disp[valid] * 100.0
    return np.clip(depth, 0, 2000).astype(np.float32)



IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 518


class BoosterDepthDataset(Dataset):
    def __init__(self, root, split="train", val_ratio=0.2, seed=42):
        self.samples = []
        self.split   = split

        scenes = sorted(os.listdir(root))
        rng    = random.Random(seed)
        rng.shuffle(scenes)
        n_val     = max(1, int(len(scenes) * val_ratio))
        val_scenes = set(scenes[:n_val])

        # Camera pairs: (camera_folder, disparity_file)
        cam_pairs = [
            ("camera_00", "disp_00.npy"),
            ("camera_02", "disp_02.npy"),
        ]

        for scene in scenes:
            is_val = scene in val_scenes
            if (split == "val") != is_val:
                continue

            calib_path = os.path.join(root, scene, "calib_00-02.xml")

            for cam_name, disp_name in cam_pairs:
                cam_dir   = os.path.join(root, scene, cam_name)
                disp_path = os.path.join(root, scene, disp_name)

                if not os.path.isdir(cam_dir) or not os.path.exists(disp_path):
                    continue

                for fname in sorted(os.listdir(cam_dir)):
                    if fname.lower().endswith((".png", ".jpg")):
                        self.samples.append((
                            os.path.join(cam_dir, fname),
                            disp_path,
                            calib_path if os.path.exists(calib_path) else None
                        ))

        print(f"[{split}] {len(self.samples)} samples from {root} "
              f"(both cameras)")

    def __len__(self):
        return len(self.samples)

    def _augment(self, img, depth):
        """Rich augmentation for training."""

        if random.random() > 0.5:
            img   = cv2.flip(img,   1)
            depth = cv2.flip(depth, 1)


        if random.random() > 0.8:
            img   = cv2.flip(img,   0)
            depth = cv2.flip(depth, 0)


        if random.random() > 0.5:
            h, w  = img.shape[:2]
            scale = random.uniform(0.8, 1.0)
            nh, nw = int(h * scale), int(w * scale)
            y = random.randint(0, h - nh)
            x = random.randint(0, w - nw)
            img   = img[y:y+nh, x:x+nw]
            depth = depth[y:y+nh, x:x+nw]
            img   = cv2.resize(img,   (w, h), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)


        if random.random() > 0.5:
            img_f = img.astype(np.float32)

            img_f *= random.uniform(0.8, 1.2)

            mean  = img_f.mean()
            img_f = (img_f - mean) * random.uniform(0.8, 1.2) + mean
            img   = np.clip(img_f, 0, 255).astype(np.uint8)


        if random.random() > 0.8:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        return img, depth

    def __getitem__(self, idx):
        img_path, disp_path, calib_path = self.samples[idx]

        img  = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        disp = np.load(disp_path).astype(np.float32)

        focal, baseline = 3757.0, 0.12
        if calib_path:
            try:
                focal, baseline = load_calibration(calib_path)
            except Exception:
                pass

        depth_cm = disp_to_depth_cm(disp, focal, baseline)


        img      = cv2.resize(img,      (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        depth_cm = cv2.resize(depth_cm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)


        if self.split == "train":
            img, depth_cm = self._augment(img, depth_cm)


        img = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        return (
            torch.tensor(img).permute(2, 0, 1).float(),
            torch.tensor(depth_cm).unsqueeze(0).float()
        )




def gradient_loss(pred, target):
    """Edge-aware gradient loss — preserves depth boundaries."""
    def gradient(x):
        gx = x[:, :, :, :-1] - x[:, :, :, 1:]
        gy = x[:, :, :-1, :] - x[:, :, 1:, :]
        return gx, gy
    pg_x, pg_y = gradient(pred)
    tg_x, tg_y = gradient(target)
    return (torch.abs(pg_x - tg_x).mean() + torch.abs(pg_y - tg_y).mean())


def depth_loss(pred, target):
    valid = (target > 0) & (target < 2000)
    if valid.sum() < 100:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)

    p = pred[valid]
    t = target[valid]


    log_diff = torch.log(p.clamp(1e-3)) - torch.log(t.clamp(1e-3))
    silog = torch.sqrt((log_diff**2).mean() - 0.5*(log_diff.mean()**2) + 1e-6)


    l1 = torch.abs(p - t).mean()


    p4 = pred.unsqueeze(0) if pred.dim() == 3 else pred
    t4 = target.unsqueeze(0) if target.dim() == 3 else target
    grad = gradient_loss(p4, t4)

    return silog + 0.1 * l1 + 0.1 * grad




def build_model(device):
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        raise ImportError(
            "Run: set PYTHONPATH=%PYTHONPATH%;Depth-Anything-V2;Depth-Anything-V2\\metric_depth"
        )

    model = DepthAnythingV2(
        encoder='vitl', features=256,
        out_channels=[256, 512, 1024, 1024]
    )

    ckpt = "checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"
    if not os.path.exists(ckpt):
        import urllib.request
        os.makedirs("checkpoints", exist_ok=True)
        print("Downloading Depth Anything V2 checkpoint (~1.3GB)...")
        urllib.request.urlretrieve(
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"
            "/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth", ckpt)
        print("Downloaded!")

    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    print(f"Depth Anything V2 loaded on {device}")
    return model


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds = BoosterDepthDataset(args.data, split="train")
    val_ds   = BoosterDepthDataset(args.data, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = build_model(device)


    for name, param in model.named_parameters():
        if "pretrained" in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )


    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler("cuda", enabled=(device == "cuda"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_rmse = float('inf')

    for epoch in range(args.epochs):

        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [train]")
        for imgs, depths in pbar:
            imgs, depths = imgs.to(device), depths.to(device)
            optimizer.zero_grad()
            with autocast("cuda", enabled=(device == "cuda")):
                pred = model(imgs)
                if pred.dim() == 3:
                    pred = pred.unsqueeze(1)
                loss = depth_loss(pred * 100.0, depths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")


        model.eval()
        rmses = []
        with torch.no_grad():
            for imgs, depths in tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs} [val]  "):
                imgs, depths = imgs.to(device), depths.to(device)
                with autocast("cuda", enabled=(device == "cuda")):
                    pred = model(imgs)
                    if pred.dim() == 3:
                        pred = pred.unsqueeze(1)
                    pred_cm = pred * 100.0
                valid = (depths > 0) & (depths < 2000)
                if valid.sum() > 0:
                    p = pred_cm[valid].float()
                    t = depths[valid].float()

                    A = torch.stack([p, torch.ones_like(p)], dim=1)
                    result = torch.linalg.lstsq(A, t.unsqueeze(1))
                    scale = result.solution[0].item()
                    shift = result.solution[1].item()
                    scale = max(0.1, min(scale, 10.0))
                    p_aligned = p * scale + shift
                    rmse = torch.sqrt(((p_aligned - t)**2).mean())
                    rmses.append(rmse.item())

        avg_loss = np.mean(losses)
        avg_rmse = np.mean(rmses) if rmses else 999.0
        cur_lr   = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | "
              f"Val RMSE: {avg_rmse:.2f} cm | LR: {cur_lr:.2e}")

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            torch.save(model.state_dict(), out_dir / "depth_best.pth")
            print(f"  Best checkpoint saved (RMSE={best_rmse:.2f} cm)")

        scheduler.step()

    torch.save(model.state_dict(), out_dir / "depth_final.pth")
    print(f"\nDone! Best RMSE: {best_rmse:.2f} cm")
    print("Now run: python infer_depth_final.py --data val_mono_nogt --train train --sam2_ckpt models/sam2_best.pth --out depth_final")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",   default="train")
    p.add_argument("--out",    default="depth_models")
    p.add_argument("--epochs", type=int,   default=50)
    p.add_argument("--batch",  type=int,   default=1)
    p.add_argument("--lr",     type=float, default=1e-4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())