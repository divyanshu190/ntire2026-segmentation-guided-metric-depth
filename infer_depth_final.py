import argparse
import os
import sys
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "Depth-Anything-V2"))
sys.path.insert(0, str(Path(__file__).parent / "Depth-Anything-V2" / "metric_depth"))



def preprocess(img):
    img_f = img.astype(np.float32)
    mg = img_f.mean()
    for c in range(3):
        img_f[:,:,c] *= mg / (img_f[:,:,c].mean() + 1e-6)
    img = np.clip(img_f, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def parse_calib(path):
    focal, baseline = 3757.0, 0.12
    try:
        for elem in ET.parse(path).getroot().iter():
            tag  = (elem.tag  or "").lower()
            name = (elem.get("name") or "").lower()
            text = (elem.text or "").strip()
            if ("p1" in tag or "p1" in name) and text:
                focal = float(text.split()[0])
            if (tag == "t" or name == "t") and text:
                baseline = abs(float(text.split()[0])) / 1000.0
    except Exception:
        pass
    return focal, baseline


def disp_to_cm(disp, focal, baseline):
    d = np.zeros_like(disp, dtype=np.float32)
    v = disp > 0
    d[v] = (focal * baseline) / disp[v] * 100.0
    return np.clip(d, 0, 2000)



def lse_align(pred, gt):
    v = (gt > 10) & (gt < 1500) & (pred > 0)
    if v.sum() < 500:
        return 1.0, 0.0
    p, g = pred[v].flatten(), gt[v].flatten()
    A = np.stack([p, np.ones_like(p)], 1)
    s, sh = np.linalg.lstsq(A, g, rcond=None)[0]
    return float(np.clip(s, 0.1, 10.0)), float(sh)


def compute_alignment(da_model, zoe_model, train_root, device):
    print("Computing LSE alignment from training disparity...")
    da_s, da_sh, zoe_s, zoe_sh = [], [], [], []

    for scene in tqdm(sorted(os.listdir(train_root))[:20], desc="Aligning"):
        cam   = os.path.join(train_root, scene, "camera_00")
        disp  = os.path.join(train_root, scene, "disp_00.npy")
        calib = os.path.join(train_root, scene, "calib_00-02.xml")
        if not os.path.isdir(cam) or not os.path.exists(disp):
            continue
        imgs = sorted([f for f in os.listdir(cam) if f.endswith(".png")])
        if not imgs:
            continue

        img = cv2.cvtColor(cv2.imread(os.path.join(cam, imgs[0])), cv2.COLOR_BGR2RGB)
        img = preprocess(img)
        H, W = img.shape[:2]

        focal, base = parse_calib(calib) if os.path.exists(calib) else (3757.0, 0.12)
        gt = disp_to_cm(np.load(disp).astype(np.float32), focal, base)


        img_s = cv2.resize(img, (int(W*0.2), int(H*0.2)), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            da_pred = da_model.infer_image(img_s) * 100.0
            da_r = cv2.resize(da_pred.astype(np.float32),
                              (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            s, sh = lse_align(da_r, gt)
            da_s.append(s); da_sh.append(sh)

            if zoe_model is not None:
                try:
                    t = torch.tensor(img_s/255.).permute(2,0,1).unsqueeze(0).float().to(device)
                    zp = zoe_model.infer(t).squeeze().cpu().numpy() * 100.0
                    zr = cv2.resize(zp.astype(np.float32),
                                    (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
                    s2, sh2 = lse_align(zr, gt)
                    zoe_s.append(s2); zoe_sh.append(sh2)
                except Exception:
                    pass

    r_da  = (float(np.median(da_s)),  float(np.median(da_sh)))  if da_s  else (1., 0.)
    r_zoe = (float(np.median(zoe_s)), float(np.median(zoe_sh))) if zoe_s else (1., 0.)
    print(f"DA  alignment: scale={r_da[0]:.4f},  shift={r_da[1]:.4f}")
    print(f"Zoe alignment: scale={r_zoe[0]:.4f}, shift={r_zoe[1]:.4f}")
    return r_da, r_zoe



def multiscale_tta_predict(model, img, scales=[0.2, 0.4, 0.6], mode="da"):
    """
    Run inference at 3 scales × 4 flips = 12 predictions total.
    Average all predictions for maximum accuracy.
    This is what top teams use for multi-scale inference.
    """
    H, W = img.shape[:2]
    all_preds = []

    for scale in scales:
        sH, sW = int(H * scale), int(W * scale)
        img_s = cv2.resize(img, (sW, sH), interpolation=cv2.INTER_AREA)


        augs = [
            img_s,
            img_s[:, ::-1].copy(),
            img_s[::-1].copy(),
            img_s[::-1, ::-1].copy(),
        ]

        for i, aug in enumerate(augs):
            with torch.no_grad():
                if mode == "da":
                    d = model.infer_image(aug)
                else:
                    t = torch.tensor(aug/255.).permute(2,0,1).unsqueeze(0).float()
                    if next(model.parameters()).is_cuda:
                        t = t.cuda()
                    d = model.infer(t).squeeze().cpu().numpy()


            if i == 1: d = d[:, ::-1].copy()
            elif i == 2: d = d[::-1].copy()
            elif i == 3: d = d[::-1, ::-1].copy()


            d_full = cv2.resize(d.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            all_preds.append(d_full)



    weights = []
    for scale in scales:
        w = 1.0 + scale
        weights.extend([w] * 4)

    weights = np.array(weights)
    weights = weights / weights.sum()

    weighted = np.zeros((H, W), dtype=np.float32)
    for pred, w in zip(all_preds, weights):
        weighted += w * pred

    return weighted



def tom_inpaint(depth, img, mask_cat_path=None):
    if mask_cat_path and os.path.exists(mask_cat_path):
        tom = cv2.imread(mask_cat_path, 0)
        tom = cv2.resize(tom, (depth.shape[1], depth.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        tom = (tom > 127).astype(np.uint8) * 255
    else:
        grad = cv2.Sobel(depth, cv2.CV_32F, 1, 1, ksize=3)
        thresh = np.percentile(np.abs(grad), 95)
        tom = (np.abs(grad) > thresh).astype(np.uint8) * 255
        tom = cv2.dilate(tom, np.ones((7,7), np.uint8))

    if tom.sum() == 0:
        return depth

    d_u16 = np.clip(depth / 2000.0 * 65535, 0, 65535).astype(np.uint16)
    inp   = cv2.inpaint(d_u16, tom, 15, cv2.INPAINT_TELEA)
    inp_f = inp.astype(np.float32) / 65535.0 * 2000.0
    alpha = (tom / 255.0).astype(np.float32)
    return ((1 - alpha) * depth + alpha * inp_f).astype(np.float32)


def sam2_sharpen(depth, img, sam2, device):
    if sam2 is None:
        return depth
    try:
        MEAN = np.array([0.485, 0.456, 0.406], np.float32)
        STD  = np.array([0.229, 0.224, 0.225], np.float32)
        ir = cv2.resize(img, (1024,1024), interpolation=cv2.INTER_AREA)
        it = torch.tensor((ir/255.-MEAN)/STD).permute(2,0,1).unsqueeze(0).float().to(device)
        with torch.no_grad():
            bo = sam2.image_encoder(it)
            ie, fpn = bo["vision_features"], bo["backbone_fpn"]
            fs1 = sam2.sam_mask_decoder.conv_s1(fpn[1])
            fs0 = sam2.sam_mask_decoder.conv_s0(fpn[0])
            sp, dp = sam2.sam_prompt_encoder(points=None, boxes=None, masks=None)
            masks,_,_,_ = sam2.sam_mask_decoder(
                image_embeddings=ie,
                image_pe=sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sp,
                dense_prompt_embeddings=dp,
                multimask_output=False, repeat_image=False,
                high_res_features=[fs0, fs1])
            mask = torch.sigmoid(masks[0,0]).cpu().numpy()

        mask_r = cv2.resize(mask, (depth.shape[1], depth.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        edges = cv2.dilate(
            cv2.Canny((mask_r*255).astype(np.uint8), 50, 150),
            np.ones((5,5), np.uint8))
        em = (edges > 0).astype(np.float32)
        ds = cv2.bilateralFilter(depth.astype(np.float32),  5,  30,  30)
        dm = cv2.bilateralFilter(depth.astype(np.float32), 15, 100, 100)
        return (em*ds + (1-em)*dm).astype(np.float32)
    except Exception as e:
        print(f"[SAM2 sharpen] {e}")
        return depth



def load_da(device):
    from depth_anything_v2.dpt import DepthAnythingV2
    m = DepthAnythingV2(encoder='vitl', features=256,
                        out_channels=[256,512,1024,1024])
    ckpt = "depth_models/depth_best.pth"
    if not os.path.exists(ckpt):
        ckpt = "checkpoints/depth_anything_v2_metric_hypersim_vitl.pth"
    if not os.path.exists(ckpt):
        import urllib.request
        os.makedirs("checkpoints", exist_ok=True)
        print("Downloading Depth Anything V2 (~1.3GB)...")
        urllib.request.urlretrieve(
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large"
            "/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth", ckpt)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    m.load_state_dict(state, strict=False)
    m.to(device).eval()
    print(f"Depth Anything V2 loaded: {ckpt}")
    return m


def load_zoe(device):
    try:
        m = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
        m.to(device).eval()
        print("ZoeDepth loaded!")
        return m
    except Exception as e:
        print(f"[ZoeDepth not available: {e}]")
        return None


def load_sam2(ckpt, device):
    if not ckpt or not os.path.exists(ckpt):
        print("[No SAM2 checkpoint — skipping SAM2 refinement]")
        return None
    try:
        import sam2 as _s
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        cfg_dir = str(Path(_s.__file__).parent / "configs" / "sam2")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=cfg_dir, version_base="1.2"):
            cfg = compose(config_name="sam2_hiera_l")
        model = instantiate(cfg.model)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        print(f"SAM2 loaded: {ckpt}")
        return model
    except Exception as e:
        print(f"[SAM2 load failed: {e}]")
        return None



def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading models...")

    da   = load_da(device)
    zoe  = load_zoe(device)
    sam2 = load_sam2(args.sam2_ckpt, device)

    # LSE alignment
    (da_s, da_sh), (zoe_s, zoe_sh) = compute_alignment(da, zoe, args.train, device) \
        if os.path.isdir(args.train) else ((1.,0.),(1.,0.))

    data_root = Path(args.data)
    out_root  = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(data_root.rglob("camera_00/*.png"))
    print(f"\nProcessing {len(img_paths)} images")
    print("Pipeline: Multi-scale(3) × TTA(4) + ZoeDepth + SAM2 + ToM inpainting")

    for img_path in tqdm(img_paths):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        img_p = preprocess(img_rgb)


        da_d = multiscale_tta_predict(da, img_p,
                                      scales=[0.2, 0.4, 0.6],
                                      mode="da") * 100.0
        da_d = np.clip(da_s * da_d + da_sh, 0, 2000)


        if zoe is not None:
            zoe_d = multiscale_tta_predict(zoe, img_p,
                                           scales=[0.2, 0.4, 0.6],
                                           mode="zoe") * 100.0
            zoe_d = np.clip(zoe_s * zoe_d + zoe_sh, 0, 2000)

            depth = 0.6 * da_d + 0.4 * zoe_d
        else:
            depth = da_d


        scene_name = img_path.parent.parent.name
        mask_cat = os.path.join(args.train, scene_name, "mask_cat.png") \
            if args.train else None
        depth = tom_inpaint(depth, img_p, mask_cat)


        depth = sam2_sharpen(depth, img_p, sam2, device)


        depth = cv2.bilateralFilter(depth.astype(np.float32), 9, 75, 75)


        depth_final = cv2.resize(depth, (orig_w, orig_h),
                                 interpolation=cv2.INTER_LINEAR).astype(np.float32)

        sp = out_root / img_path.parent.parent.name / f"{img_path.stem}.npy"
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(sp), depth_final)

    print(f"\nSaved to: {out_root}")

    # Zip
    zip_path = Path(args.out + "_submission.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(out_root.rglob("*.npy")):
            zf.write(f, f.relative_to(out_root))
    print(f"Submission zip: {zip_path}")
    print("Upload to CodaLab Test phase!")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",      default="val_mono_nogt")
    p.add_argument("--train",     default="train")
    p.add_argument("--sam2_ckpt", default="models/sam2_best.pth")
    p.add_argument("--out",       default="depth_final")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())