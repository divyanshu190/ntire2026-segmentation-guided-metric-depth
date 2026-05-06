

import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten spatial dims
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union        = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
   

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t   = targets * probs + (1 - targets) * (1 - probs)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class IoULoss(nn.Module):
    

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        total        = (probs + targets).sum(dim=1)
        union        = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()


class CombinedSegmentationLoss(nn.Module):
   

    def __init__(
        self,
        w_bce:   float = 0.3,
        w_dice:  float = 0.4,
        w_focal: float = 0.2,
        w_iou:   float = 0.1,
    ):
        super().__init__()
        self.w_bce   = w_bce
        self.w_dice  = w_dice
        self.w_focal = w_focal
        self.w_iou   = w_iou

        self.dice  = DiceLoss()
        self.focal = FocalLoss()
        self.iou   = IoULoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce   = F.binary_cross_entropy_with_logits(logits, targets)
        dice  = self.dice(logits,  targets)
        focal = self.focal(logits, targets)
        iou   = self.iou(logits,   targets)

        loss = (
            self.w_bce   * bce
            + self.w_dice  * dice
            + self.w_focal * focal
            + self.w_iou   * iou
        )
        return loss, {
            "bce": bce.item(),
            "dice": dice.item(),
            "focal": focal.item(),
            "iou": iou.item(),
        }



_loss_fn = CombinedSegmentationLoss()

def segmentation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total, _ = _loss_fn(pred, target)
    return total
