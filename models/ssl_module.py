"""
Loss functions and label-mixing augmentations.

We keep the "ssl_module" filename for backward compatibility with the original
folder structure, but the contents have been swapped for things that actually
move accuracy: focal loss, class weighting, mixup, cutmix.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Focal loss                                                                 #
# --------------------------------------------------------------------------- #

class FocalLoss(nn.Module):
    """
    Multi-class focal loss with optional class weights and label smoothing.

    L = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        # Cross-entropy with label smoothing — gives us log p
        ce = F.cross_entropy(
            logits, target,
            weight=self.weight, label_smoothing=self.label_smoothing,
            reduction="none",
        )
        # focal modulation uses true-class probability
        with torch.no_grad():
            pt = torch.softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        focal_term = (1.0 - pt).clamp(min=1e-8) ** self.gamma
        loss = focal_term * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(cfg, class_weights: torch.Tensor = None):
    weight = class_weights if cfg.USE_CLASS_WEIGHTS else None
    if cfg.USE_FOCAL_LOSS:
        return FocalLoss(
            gamma=cfg.FOCAL_GAMMA,
            weight=weight,
            label_smoothing=cfg.LABEL_SMOOTHING,
        )
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.LABEL_SMOOTHING)


# --------------------------------------------------------------------------- #
#  MixUp / CutMix                                                             #
# --------------------------------------------------------------------------- #

def mixup_data(x, y, alpha: float = 0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], float(lam)


def cutmix_data(x, y, alpha: float = 1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)

    B, C, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    # adjust lam to the *actual* area replaced
    lam_adj = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
    return x_mixed, y, y[idx], float(lam_adj)


def mixed_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
