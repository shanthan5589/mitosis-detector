"""
STEP 4 OF 2-STAGE PIPELINE: Train Stage 2 EfficientNet-B2 classifier.

What this script does
---------------------
Trains a binary classifier (mitotic vs not-mitotic) on the crops produced by
prepare_stage2_data.py.

Unlike the original single-stage approach, the training data here consists of:
  - True positives (real mitotic crops that YOLO correctly detected)
  - Hard negatives (non-mitotic crops that confused YOLO)
  - False negatives (mitotic crops that YOLO missed)

This focused dataset forces the model to learn the genuinely hard distinctions.

Model: EfficientNet-B2 (pretrained on ImageNet), final layer replaced with Linear(1408 → 1).
Loss:  BCEWithLogitsLoss with pos_weight to handle class imbalance.
Metric for best-model selection: validation F1 on the positive (mitotic) class.

Prerequisites
-------------
  Run prepare_stage2_data.py first to generate stage2_data/.

Usage
-----
    python stage2_classifier/train.py
    (run from the repo root: D:/Github/mitosis-detector)

Output
------
    models/stage2_best.pth    ← best checkpoint (highest val F1)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    STAGE2_DATA_DIR,
    MODELS_DIR,
    STAGE2_IMG_SIZE,
    STAGE2_EPOCHS,
    STAGE2_BATCH,
    STAGE2_LR,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CropDataset(Dataset):
    """
    Loads crop images and binary labels from a CSV manifest.
    CSV format: path (absolute), label (0 or 1)
    """

    def __init__(self, csv_path: Path, transform):
        df = pd.read_csv(csv_path)
        self.paths  = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# ImageNet mean/std — EfficientNet was pretrained with these
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((STAGE2_IMG_SIZE, STAGE2_IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        # Colour jitter handles staining variation across slides
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((STAGE2_IMG_SIZE, STAGE2_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    """
    EfficientNet-B2 with the default classifier head replaced by a single
    binary output neuron. B2 feature dimension is 1408.
    """
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features   # 1408
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0
    use_amp    = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)   # (B,) → (B, 1) for BCEWithLogitsLoss

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds       = (torch.sigmoid(logits.detach().float()) > 0.5).float()
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device) -> tuple[float, float, float]:
    """Returns (val_loss, val_accuracy, val_f1_positive_class)."""
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    use_amp    = device.type == "cuda"

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss   = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(logits.float()) > 0.5).float()
            all_preds.extend(preds.cpu().squeeze(1).tolist())
            all_labels.extend(labels.cpu().squeeze(1).tolist())

    val_loss = total_loss / len(all_labels)
    val_acc  = accuracy_score(all_labels, all_preds)
    val_f1   = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    return val_loss, val_acc, val_f1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    print(f"Device: {device}  |  AMP: {device.type == 'cuda'}\n")

    # ── Load manifests ────────────────────────────────────────────────────
    train_csv = STAGE2_DATA_DIR / "train" / "train.csv"
    val_csv   = STAGE2_DATA_DIR / "val"   / "val.csv"
    for p in [train_csv, val_csv]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run prepare_stage2_data.py first.")

    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)

    if len(train_df) == 0:
        raise RuntimeError(f"Training manifest is empty: {train_csv}. Re-run prepare_stage2_data.py.")
    if len(val_df) == 0:
        raise RuntimeError(f"Validation manifest is empty: {val_csv}. Re-run prepare_stage2_data.py.")

    n_pos = (train_df["label"] == 1).sum()
    n_neg = (train_df["label"] == 0).sum()
    if n_pos == 0:
        raise RuntimeError("No positive (mitotic) crops in training set. Check YOLO recall or BOX_SIZE.")
    if n_neg == 0:
        raise RuntimeError("No negative crops in training set. Check YOLO_CONF_THRESH.")

    print(f"Train: {n_pos} mitotic  |  {n_neg} non-mitotic")
    print(f"Val:   {(val_df['label']==1).sum()} mitotic  |  {(val_df['label']==0).sum()} non-mitotic\n")

    # ── Datasets and loaders ──────────────────────────────────────────────
    train_ds = CropDataset(train_csv, get_train_transform())
    val_ds   = CropDataset(val_csv,   get_val_transform())

    train_loader = DataLoader(train_ds, batch_size=STAGE2_BATCH, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=STAGE2_BATCH, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── Model, loss, optimiser ────────────────────────────────────────────
    model = build_model().to(device)

    # pos_weight compensates for class imbalance.
    # If there are 3x more negatives than positives, we weight positive errors 3x higher.
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"pos_weight = {pos_weight.item():.3f}\n")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=STAGE2_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Training loop ─────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_path = MODELS_DIR / "stage2_best.pth"
    best_f1   = -1.0

    for epoch in range(1, STAGE2_EPOCHS + 1):
        tr_loss, tr_acc           = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{STAGE2_EPOCHS}  "
            f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  val_F1={val_f1:.4f}  "
            f"lr={lr_now:.2e}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best F1 = {best_f1:.4f} — saved to {best_path}")

    print(f"\nTraining complete. Best val F1 = {best_f1:.4f}")
    print(f"Best weights: {best_path}")
    print("\nNext step: run evaluate.py")


if __name__ == "__main__":
    main()
