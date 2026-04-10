"""Custom IoU loss 
"""
import torch
import torch.nn as nn


class IoULoss(nn.Module):
    """IoU loss for bounding box regression.
    
    Computes 1 - IoU between predicted and target bounding boxes.
    IoU (Intersection over Union) measures the overlap between two boxes.
    Using 1 - IoU as a loss drives the model to maximize overlap.
    
    Unlike L1/L2 losses on raw coordinates, IoU loss is:
    - Scale invariant (same loss for small and large boxes with same overlap)
    - Directly optimizes the evaluation metric
    - Numerically stable with the eps guard
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """
        Initialize the IoULoss module.
        Args:
            eps: Small value to avoid division by zero.
            reduction: Specifies the reduction to apply to the output: 'mean' | 'sum' | 'none'.
        """
        super().__init__()
        self.eps = eps

        # Validate reduction
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """Compute IoU loss between predicted and target bounding boxes.

        Args:
            pred_boxes:   [B, 4] predicted boxes in (x_center, y_center, width, height) format.
            target_boxes: [B, 4] target boxes   in (x_center, y_center, width, height) format.

        Returns:
            Scalar IoU loss (if reduction is 'mean' or 'sum'),
            or per-sample losses of shape [B] (if reduction is 'none').
        """
        # ── Convert (cx, cy, w, h) → (x1, y1, x2, y2) ──────────────────────
        # Predicted box corners
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2   # cx - w/2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2   # cy - h/2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2   # cx + w/2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2   # cy + h/2

        # Target box corners
        tgt_x1  = target_boxes[:, 0] - target_boxes[:, 2] / 2
        tgt_y1  = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tgt_x2  = target_boxes[:, 0] + target_boxes[:, 2] / 2
        tgt_y2  = target_boxes[:, 1] + target_boxes[:, 3] / 2

        # ── Intersection ─────────────────────────────────────────────────────
        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        # Clamp to 0 — no intersection if boxes don't overlap
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        intersection = inter_w * inter_h                     # [B]

        # ── Union ─────────────────────────────────────────────────────────────
        pred_area   = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        target_area = (tgt_x2  - tgt_x1).clamp(min=0) * (tgt_y2  - tgt_y1).clamp(min=0)
        union = pred_area + target_area - intersection + self.eps  # [B]

        # ── IoU and loss ──────────────────────────────────────────────────────
        iou  = intersection / union                          # [B], in [0, 1]
        loss = 1.0 - iou                                     # [B], lower is better

        # ── Reduction ─────────────────────────────────────────────────────────
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss
