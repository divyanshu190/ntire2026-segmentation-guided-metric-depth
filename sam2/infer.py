

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm



MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 1024


def preprocess(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, (orig_h, orig_w)


def build_model(ckpt_path, device):
    import sam2 as _sam2_pkg
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    config_dir = str(Path(_sam2_pkg.__file__).parent / "configs" / "sam2")
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.2"):
        cfg = compose(config_name="sam2_hiera_l")

    model = instantiate(cfg.model)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded from {ckpt_path}")
    return model


@torch.no_grad()
def predict(model, img_tensor, device):
    img_tensor = img_tensor.to(device)
    dec = model.sam_mask_decoder

    backbone_out   = model.image_encoder(img_tensor)
    img_embeddings = backbone_out["vision_features"]
    fpn            = backbone_out["backbone_fpn"]

    feat_s1 = dec.conv_s1(fpn[1])
    feat_s0 = dec.conv_s0(fpn[0])
    high_res_features = [feat_s0, feat_s1]

    sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
        points=None, boxes=None, masks=None
    )

    low_res_masks, _, _, _ = dec(
        image_embeddings=img_embeddings,
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )

    mask = torch.sigmoid(low_res_masks[0, 0])   
    return mask.cpu().numpy()


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_model(args.ckpt, device)

    data_root = Path(args.data)
    out_root  = Path(args.out)


    img_paths = sorted(data_root.rglob("camera_00/*.png"))
    print(f"Found {len(img_paths)} images")

    for img_path in tqdm(img_paths, desc="Running inference"):
        img_tensor, (orig_h, orig_w) = preprocess(img_path)

        mask = predict(model, img_tensor, device)


        mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


        binary = (mask_resized > args.threshold).astype(np.uint8) * 255


        rel_path = img_path.relative_to(data_root)
        save_path = out_root / rel_path.parent.parent / (rel_path.stem + "_mask.png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), binary)


        orig_img = cv2.imread(str(img_path))
        overlay  = orig_img.copy()
        overlay[binary == 255] = (overlay[binary == 255] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        overlay_path = out_root / rel_path.parent.parent / (rel_path.stem + "_overlay.png")
        cv2.imwrite(str(overlay_path), overlay)

    print(f"\nDone! Masks saved to: {out_root}")
    print("Each image has two outputs:")
    print("  *_mask.png    — binary segmentation mask")
    print("  *_overlay.png — original image with mask overlaid in green")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      default="val_mono_nogt",    help="Validation folder")
    p.add_argument("--ckpt",      default="models/sam2_best.pth", help="Trained checkpoint")
    p.add_argument("--out",       default="predictions",      help="Output folder")
    p.add_argument("--threshold", type=float, default=0.5,    help="Mask threshold")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
