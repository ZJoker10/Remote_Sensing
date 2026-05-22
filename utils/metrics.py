# utils/metrics.py
# Segmentation metrics: IoU (Jaccard), Dice, Precision, Recall, F1.
# All functions accept raw logits or binary predictions.

import torch
import torch.nn.functional as F
import numpy as np


def sigmoid_threshold(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convert raw logits → binary mask."""
    return (torch.sigmoid(logits) > threshold).float()


# ─────────────────────────────────────────────────────────────────────────────
# Per-batch metrics (differentiable versions for training monitoring)
# ─────────────────────────────────────────────────────────────────────────────

def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Dice similarity coefficient (DSC / F1 over areas).
    Inputs: binary tensors of same shape, values in {0, 1}.
    """
    pred   = pred.float().view(-1)
    target = target.float().view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Intersection-over-Union (Jaccard index).
    """
    pred   = pred.float().view(-1)
    target = target.float().view(-1)
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision_recall(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6):
    """Returns (precision, recall) tuple."""
    pred   = pred.float().view(-1)
    target = target.float().view(-1)
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    precision = (tp + smooth) / (tp + fp + smooth)
    recall    = (tp + smooth) / (tp + fn + smooth)
    return precision.item(), recall.item()


# ─────────────────────────────────────────────────────────────────────────────
# Epoch-level metric accumulator
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationMetrics:
    """
    Accumulates TP, FP, FN counts across an entire epoch, then computes
    macro-averaged IoU and F1 at the end.

    Usage:
        metrics = SegmentationMetrics()
        for batch in loader:
            pred_bin = sigmoid_threshold(model(batch["image"]))
            metrics.update(pred_bin, batch["mask"])
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits:  (B, 1, H, W) raw model outputs
            targets: (B, 1, H, W) binary ground-truth masks
        """
        preds = (torch.sigmoid(logits) > self.threshold).float()
        preds   = preds.view(-1).cpu()
        targets = targets.view(-1).cpu().float()

        self.tp += (preds * targets).sum().item()
        self.fp += (preds * (1 - targets)).sum().item()
        self.fn += ((1 - preds) * targets).sum().item()

    def compute(self) -> dict:
        smooth = 1e-6
        precision = (self.tp + smooth) / (self.tp + self.fp + smooth)
        recall    = (self.tp + smooth) / (self.tp + self.fn + smooth)
        f1        = 2 * precision * recall / (precision + recall + smooth)
        iou       = (self.tp + smooth) / (self.tp + self.fp + self.fn + smooth)

        return {
            "iou":       round(iou, 4),
            "f1":        round(f1, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(torch.nn.Module):
    """1 - Dice, for use as a training loss."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return 1.0 - dice_score(probs, targets, self.smooth)


class ComboLoss(torch.nn.Module):
    """
    Weighted sum of Binary Cross-Entropy and Dice Loss.
    BCE drives per-pixel accuracy; Dice drives region overlap.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.bce  = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.bce_weight  * self.bce(logits, targets) +
                self.dice_weight * self.dice(logits, targets))


def build_loss(cfg: dict) -> torch.nn.Module:
    """Instantiate loss from config."""
    loss_type = cfg["loss"]["type"]
    if loss_type == "combo":
        return ComboLoss(cfg["loss"]["bce_weight"], cfg["loss"]["dice_weight"])
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "bce":
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
