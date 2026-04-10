"""Training entrypoint
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(model, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Checkpoint saved → {path}")


def dice_score(pred_mask: torch.Tensor, true_mask: torch.Tensor, num_classes: int = 3) -> float:
    """Compute mean Dice score across classes."""
    pred = pred_mask.argmax(dim=1)  # [B, H, W]
    dice = 0.0
    for c in range(num_classes):
        p = (pred == c).float()
        t = (true_mask == c).float()
        intersection = (p * t).sum()
        dice += (2 * intersection + 1e-6) / (p.sum() + t.sum() + 1e-6)
    return (dice / num_classes).item()


# ── Task 1: Classification ────────────────────────────────────────────────────

def train_classifier(args):
    device = get_device()
    wandb.init(project="da6401-assignment2", name="classifier", config=vars(args))

    # Data
    dataset = OxfordIIITPetDataset(root=args.data_root, split='trainval', image_size=224)
    val_size = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = VGG11Classifier(num_classes=37, dropout_p=args.dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        train_acc  = correct / total
        train_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch['image'].to(device)
                labels = batch['label'].to(device)
                logits = model(imgs)
                val_loss    += criterion(logits, labels).item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc  = val_correct / val_total
        val_loss = val_loss / len(val_loader)

        scheduler.step()

        wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc,
                   'val_loss': val_loss, 'val_acc': val_acc})
        print(f"Epoch {epoch+1}/{args.epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, 'checkpoints/classifier.pth')

    wandb.finish()


# ── Task 2: Localization ──────────────────────────────────────────────────────

def train_localizer(args):
    device = get_device()
    wandb.init(project="da6401-assignment2", name="localizer", config=vars(args))

    dataset = OxfordIIITPetDataset(root=args.data_root, split='trainval', image_size=224)
    val_size = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = VGG11Localizer(dropout_p=args.dropout_p, image_size=224).to(device)

    # Optionally load pretrained encoder from classifier
    if os.path.exists('checkpoints/classifier.pth'):
        clf_state = torch.load('checkpoints/classifier.pth', map_location=device)
        encoder_state = {k.replace('encoder.', ''): v
                         for k, v in clf_state.items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(encoder_state)
        print("Loaded pretrained encoder from classifier.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = IoULoss(reduction='mean')

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            imgs   = batch['image'].to(device)
            bboxes = batch['bbox'].to(device)
            optimizer.zero_grad()
            preds  = model(imgs)
            loss   = criterion(preds, bboxes)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch['image'].to(device)
                bboxes = batch['bbox'].to(device)
                preds  = model(imgs)
                val_loss += criterion(preds, bboxes).item()

        train_loss = total_loss / len(train_loader)
        val_loss   = val_loss   / len(val_loader)

        scheduler.step()
        wandb.log({'epoch': epoch+1, 'train_iou_loss': train_loss, 'val_iou_loss': val_loss})
        print(f"Epoch {epoch+1}/{args.epochs} | Train IoU Loss: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, 'checkpoints/localizer.pth')

    wandb.finish()


# ── Task 3: Segmentation ──────────────────────────────────────────────────────

def train_segmentation(args):
    device = get_device()
    wandb.init(project="da6401-assignment2", name="segmentation", config=vars(args))

    dataset = OxfordIIITPetDataset(root=args.data_root, split='trainval', image_size=224)
    val_size = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout_p).to(device)

    # Load pretrained encoder
    if os.path.exists('checkpoints/classifier.pth'):
        clf_state = torch.load('checkpoints/classifier.pth', map_location=device)
        encoder_state = {k.replace('encoder.', ''): v
                         for k, v in clf_state.items() if k.startswith('encoder.')}
        model.encoder.load_state_dict(encoder_state)
        print("Loaded pretrained encoder from classifier.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_dice = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            imgs  = batch['image'].to(device)
            masks = batch['mask'].to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss, val_dice = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch['image'].to(device)
                masks = batch['mask'].to(device)
                logits = model(imgs)
                val_loss += criterion(logits, masks).item()
                val_dice += dice_score(logits, masks)

        train_loss = total_loss / len(train_loader)
        val_loss   = val_loss   / len(val_loader)
        val_dice   = val_dice   / len(val_loader)

        scheduler.step()
        wandb.log({'epoch': epoch+1, 'train_loss': train_loss,
                   'val_loss': val_loss, 'val_dice': val_dice})
        print(f"Epoch {epoch+1}/{args.epochs} | Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, 'checkpoints/unet.pth')

    wandb.finish()


# ── Task 4: Multi-task ────────────────────────────────────────────────────────

def train_multitask(args):
    device = get_device()
    wandb.init(project="da6401-assignment2", name="multitask", config=vars(args))

    dataset = OxfordIIITPetDataset(root=args.data_root, split='trainval', image_size=224)
    val_size = int(0.1 * len(dataset))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = MultiTaskPerceptionModel(
        classifier_path='checkpoints/classifier.pth',
        localizer_path='checkpoints/localizer.pth',
        unet_path='checkpoints/unet.pth',
    ).to(device)

    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    cls_loss   = nn.CrossEntropyLoss()
    bbox_loss  = IoULoss(reduction='mean')
    seg_loss   = nn.CrossEntropyLoss()

    # Loss weights — balance the three tasks
    w_cls, w_bbox, w_seg = 1.0, 1.0, 1.0

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            imgs   = batch['image'].to(device)
            labels = batch['label'].to(device)
            bboxes = batch['bbox'].to(device)
            masks  = batch['mask'].to(device)

            optimizer.zero_grad()
            out = model(imgs)

            loss = (w_cls  * cls_loss(out['classification'], labels) +
                    w_bbox * bbox_loss(out['localization'], bboxes)  +
                    w_seg  * seg_loss(out['segmentation'], masks))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_total = 0
        val_acc, val_dice_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch['image'].to(device)
                labels = batch['label'].to(device)
                bboxes = batch['bbox'].to(device)
                masks  = batch['mask'].to(device)
                out = model(imgs)
                val_total += (w_cls  * cls_loss(out['classification'], labels) +
                              w_bbox * bbox_loss(out['localization'], bboxes)  +
                              w_seg  * seg_loss(out['segmentation'], masks)).item()
                val_acc        += (out['classification'].argmax(1) == labels).float().mean().item()
                val_dice_total += dice_score(out['segmentation'], masks)

        train_loss = total_loss / len(train_loader)
        val_loss   = val_total  / len(val_loader)
        val_acc    = val_acc    / len(val_loader)
        val_dice   = val_dice_total / len(val_loader)

        scheduler.step()
        wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss,
                   'val_acc': val_acc, 'val_dice': val_dice})
        print(f"Epoch {epoch+1}/{args.epochs} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Dice: {val_dice:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, 'checkpoints/multitask.pth')

    wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DA6401 Assignment 2 models')
    parser.add_argument('--task',         type=str,   default='classifier',
                        choices=['classifier', 'localizer', 'segmentation', 'multitask'])
    parser.add_argument('--data_root',    type=str,   default='data')
    parser.add_argument('--epochs',       type=int,   default=20)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--dropout_p',    type=float, default=0.5)
    args = parser.parse_args()

    tasks = {
        'classifier':   train_classifier,
        'localizer':    train_localizer,
        'segmentation': train_segmentation,
        'multitask':    train_multitask,
    }
    tasks[args.task](args)
