# Segmentation-Guided Metric Depth Estimation for Specular and Transparent Surfaces

> **NTIRE 2026 Challenge — Track 2: Metric Mono | 🏆 4th Place out of 60 teams**

A parameter-efficient pipeline for high-resolution metric depth estimation on indoor scenes containing transparent glass and mirror surfaces. The system combines a fine-tuned SAM2 segmentation model with a Depth Anything V2 + ZoeDepth ensemble, using segmentation boundaries to guide and sharpen depth predictions at challenging non-Lambertian surfaces.

---
## Result

| Input Image vs. Predicted Depth Map |
|:-----------------------------------:|
| ![Depth Estimation Result](https://github.com/user-attachments/assets/dc63811a-29fb-48e6-90c9-c5747791b2e6) |

> Blue/purple = closer objects. Orange/yellow = farther surfaces. Object boundaries are sharp and physically correct.
---

## Key Results

| Metric | Value |
|--------|-------|
| Competition | NTIRE 2026 Track 2: Metric Mono |
| Final Rank | **4th / 60 teams** |
| Total Submissions | 327 |
| Best Val RMSE (Depth) | **5.62 cm** |
| Best Val IoU (SAM2) | **0.8777** |
| Evaluation Dataset | Booster — 159 test images, 18 scenes |

---

## Method Overview

The pipeline has two independently trained components that are combined at inference time.

### Component 1 — SAM2 Segmentation Fine-Tuning

SAM2 Large was fine-tuned on the Booster training set (38 scenes, camera_00 only — 114 images). Two separate models were trained for different mask types:

- `mask_00` — scene geometry masks
- `mask_cat` — clean object boundary masks

Only the mask decoder was fine-tuned using **LoRA (rank=4)**, keeping 97.2% of parameters frozen.

| Detail | Value |
|--------|-------|
| Base model | SAM2 Large (ViT-H) |
| Frozen | Image encoder + memory encoder |
| Trainable | Mask decoder via LoRA rank=4 |
| Trainable params | 6.3M / 224M (2.8%) |
| Training images | 114 (188 train / 40 val after augmentation) |
| Epochs | 60 |
| Best val IoU | 0.8777 |

**Loss:**
```
L = 0.3 * BCE + 0.4 * Dice + 0.2 * Focal(α=0.8, γ=2) + 0.1 * IoU
```

### Component 2 — Metric Depth Estimation

Depth Anything V2 ViT-L was fine-tuned on both stereo cameras (camera_00 + camera_02) from the Booster training set — 376 training samples total. Only the DPT decoder was trained; the ViT-L encoder stayed frozen.

Ground truth depth was derived from disparity maps using stereo calibration:
```
depth_cm = (focal_length × baseline / disparity) × 100
```

| Detail | Value |
|--------|-------|
| Primary model | Depth Anything V2 ViT-L (fine-tuned) |
| Secondary model | ZoeDepth NK (pretrained only) |
| Training samples | 376 (both cameras) |
| Frozen | ViT-L encoder (200M params) |
| Trainable | DPT decoder (30.9M, 9.2%) |
| Epochs | 50 |
| Best val RMSE | 5.62 cm |

**Loss:**
```
L = SILog + 0.1 * L1 + 0.1 * Gradient
```

**Training progression:**

| Epoch | Train Loss | Val RMSE (cm) |
|-------|-----------|---------------|
| 1  | 21.23 | 21.67 |
| 5  | 3.72  | 9.72  |
| 10 | 1.97  | 6.79  |
| 19 | 1.30  | 6.14  |
| 50 | 0.53  | 5.62  |

### Inference Pipeline

```
Input Image
    ↓
Gray World + CLAHE preprocessing
    ↓
3 scales (0.2×, 0.4×, 0.6×) × 4 flips = 12 DA V2 predictions
3 scales (0.2×, 0.4×, 0.6×) × 4 flips = 12 ZoeDepth predictions
    ↓
Soft ensemble: 0.6 × DA_V2 + 0.4 × ZoeDepth
    ↓
LSE scale+shift alignment from training disparity
    ↓
ToM inpainting (TELEA) on transparent/mirror regions
    ↓
SAM2 boundary sharpening
    ↓
Bilateral filter smoothing
    ↓
Output: float32 .npy, 3008×4112, unit: cm
```

---

## Repository Structure

```
ntire2026-segmentation-guided-metric-depth/
├── sam2/
│   ├── train.py              # SAM2 fine-tuning
│   ├── dataset.py            # Booster dataset loader
│   ├── loss.py               # Combined segmentation loss
│   └── infer.py              # SAM2 inference
├── depth_estimation/
│   ├── train_depth.py        # Depth Anything V2 fine-tuning
│   └── infer_depth_final.py  # Full inference pipeline
├── Requirements.txt
└── README.md
```

---

## Dataset

**Booster Dataset** — [CVPR 2022](https://cvpr.thecvf.com/virtual/2022/paper/2198)

- 419 high-resolution stereo pairs across 64 indoor scenes
- 38 scenes (228 stereo pairs) released for training
- Image resolution: 4112 × 3008 pixels
- Contains transparent, specular, and diffuse surface annotations

---

## Setup

```bash
git clone https://github.com/divyanshu190/ntire2026-segmentation-guided-metric-depth
cd ntire2026-segmentation-guided-metric-depth
pip install -r Requirements.txt

# Clone Depth Anything V2
git clone https://github.com/DepthAnything/Depth-Anything-V2
pip install -r Depth-Anything-V2/requirements.txt
pip install timm==0.6.13
```

---

## Training

**SAM2 segmentation:**
```bash
cd sam2
python train.py --data train --epochs 60 --mask_type mask_00
```

**Depth model:**
```bash
python depth_estimation/train_depth.py \
    --data train \
    --epochs 50 \
    --batch 1 \
    --lr 1e-4 \
    --out depth_models
```

---

## Inference

```bash
python depth_estimation/infer_depth_final.py \
    --data test_mono_nogt \
    --train train \
    --sam2_ckpt models/sam2_best.pth \
    --out depth_final
```

This creates `depth_final_submission.zip` ready for CodaLab upload.

---

## Models Used

| Model | Purpose | Source |
|-------|---------|--------|
| SAM2 Large | Segmentation fine-tuning | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) |
| Depth Anything V2 ViT-L | Primary depth model | [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) |
| ZoeDepth NK | Secondary depth model | [isl-org/ZoeDepth](https://github.com/isl-org/ZoeDepth) |

---

## References

```
@inproceedings{booster,
  title={Open Challenges in Deep Stereo: The Booster Dataset},
  author={Ramirez et al.},
  booktitle={CVPR},
  year={2022}
}

@article{dav2,
  title={Depth Anything V2},
  author={Yang et al.},
  journal={arXiv},
  year={2024}
}

@article{zoedepth,
  title={ZoeDepth: Zero-shot Transfer by Combining Relative and Metric Depth},
  author={Bhat et al.},
  journal={arXiv},
  year={2023}
}

@article{sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi et al.},
  journal={arXiv},
  year={2024}
}
```

---

## Competition

- **Challenge:** [NTIRE 2026: HR Depth from Images of Specular and Transparent Surfaces](https://codabench.org/competitions/12778)
- **Track:** Track 2 — Metric Mono
- **Result:** 4th place / 60 teams
