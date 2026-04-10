"""Inference and evaluation
"""
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path: str, device: torch.device) -> MultiTaskPerceptionModel:
    model = MultiTaskPerceptionModel(
        classifier_path='checkpoints/classifier.pth',
        localizer_path='checkpoints/localizer.pth',
        unet_path='checkpoints/unet.pth',
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized image tensor back to uint8 numpy array."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def draw_bbox(img_np: np.ndarray, bbox: torch.Tensor, color: str) -> Image.Image:
    """Draw a bounding box on a numpy image. bbox = [cx, cy, w, h] in pixels."""
    img_pil = Image.fromarray(img_np)
    draw    = ImageDraw.Draw(img_pil)
    cx, cy, w, h = bbox.tolist()
    x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    return img_pil


def compute_iou(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute IoU for a single box pair."""
    px1 = pred[0] - pred[2]/2;   py1 = pred[1] - pred[3]/2
    px2 = pred[0] + pred[2]/2;   py2 = pred[1] + pred[3]/2
    tx1 = target[0] - target[2]/2; ty1 = target[1] - target[3]/2
    tx2 = target[0] + target[2]/2; ty2 = target[1] + target[3]/2
    ix1 = max(px1, tx1); iy1 = max(py1, ty1)
    ix2 = min(px2, tx2); iy2 = min(py2, ty2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    pred_area   = (px2-px1) * (py2-py1)
    target_area = (tx2-tx1) * (ty2-ty1)
    union = pred_area + target_area - inter + eps
    return inter / union


def dice_score(pred_logits: torch.Tensor, true_mask: torch.Tensor, num_classes: int = 3) -> float:
    pred = pred_logits.argmax(dim=1)
    dice = 0.0
    for c in range(num_classes):
        p = (pred == c).float()
        t = (true_mask == c).float()
        intersection = (p * t).sum()
        dice += (2 * intersection + 1e-6) / (p.sum() + t.sum() + 1e-6)
    return (dice / num_classes).item()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(args):
    device = get_device()
    wandb.init(project="da6401-assignment2", name="inference-evaluation")

    # Dataset
    test_ds     = OxfordIIITPetDataset(root=args.data_root, split='test', image_size=224)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model = load_model(args.checkpoint, device)

    # Metrics accumulators
    all_labels, all_preds = [], []
    total_dice, total_iou = 0.0, 0.0
    n_batches = 0

    # W&B table for bbox visualization (10 samples)
    bbox_table = wandb.Table(columns=["image", "iou", "confidence", "result"])
    seg_table  = wandb.Table(columns=["original", "ground_truth", "prediction"])
    bbox_samples_logged = 0
    seg_samples_logged  = 0

    with torch.no_grad():
        for batch in test_loader:
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)
            bboxes = batch['bbox'].to(device)
            masks  = batch['mask'].to(device)

            out = model(imgs)

            # Classification
            preds = out['classification'].argmax(1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Segmentation dice
            total_dice += dice_score(out['segmentation'], masks)

            # IoU per sample
            for i in range(imgs.size(0)):
                iou_val = compute_iou(out['localization'][i].cpu(), bboxes[i].cpu())
                total_iou += iou_val

                # Log 10 bbox images to W&B
                if bbox_samples_logged < 10:
                    img_np  = denormalize(imgs[i])
                    img_gt  = draw_bbox(img_np,  bboxes[i].cpu(),             color='green')
                    img_pred = draw_bbox(np.array(img_gt), out['localization'][i].cpu(), color='red')
                    conf    = float(out['classification'][i].softmax(0).max().item())
                    bbox_table.add_data(wandb.Image(img_pred), round(iou_val, 4), round(conf, 4),
                                        'good' if iou_val > 0.5 else 'failure')
                    bbox_samples_logged += 1

                # Log 5 segmentation images to W&B
                if seg_samples_logged < 5:
                    img_np    = denormalize(imgs[i])
                    gt_mask   = masks[i].cpu().numpy().astype(np.uint8) * 127
                    pred_mask = out['segmentation'][i].argmax(0).cpu().numpy().astype(np.uint8) * 127
                    seg_table.add_data(
                        wandb.Image(img_np),
                        wandb.Image(gt_mask),
                        wandb.Image(pred_mask)
                    )
                    seg_samples_logged += 1

            n_batches += 1

    # Final metrics
    f1  = f1_score(all_labels, all_preds, average='macro')
    avg_dice = total_dice / n_batches
    avg_iou  = total_iou  / len(test_ds)

    print(f"\n── Test Results ──────────────────────────")
    print(f"  Macro F1 Score : {f1:.4f}")
    print(f"  Mean IoU       : {avg_iou:.4f}")
    print(f"  Mean Dice Score: {avg_dice:.4f}")

    wandb.log({
        "test_macro_f1":  f1,
        "test_mean_iou":  avg_iou,
        "test_mean_dice": avg_dice,
        "bbox_predictions": bbox_table,
        "segmentation_results": seg_table,
    })
    wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on DA6401 Assignment 2 pipeline')
    parser.add_argument('--checkpoint',  type=str, default='checkpoints/multitask.pth')
    parser.add_argument('--data_root',   type=str, default='data')
    parser.add_argument('--batch_size',  type=int, default=16)
    args = parser.parse_args()
    evaluate(args)
