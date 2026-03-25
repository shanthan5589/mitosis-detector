"""
Training script: Loads clean patches, splits by slide, trains ResNet-18
with transfer learning, evaluates on validation set each epoch.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score

import cv2
import pandas as pd
from collections import defaultdict

from config import TRAINING_DATA_DIR, TRAIN_META_DATA_DIR, CLEAN_PATHS_FILE

from model_utils import TRAIN_SLIDES, VAL_SLIDES

from model_utils import (
    build_model,
    NUM_EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    get_train_transform,
    get_eval_transform,
    NUM_WORKERS,
    PIN_MEMORY,
)

class MitosisDataset(Dataset):

    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}. File may be missing or corrupted.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_clean_paths():
    clean_paths = defaultdict(list)

    with open(CLEAN_PATHS_FILE, "r") as f:
        for line in f:
            label, filename = line.strip().split(",")
            subdir = "mitotic" if label == "2" else "non_mitotic"
            clean_paths[label].append(TRAINING_DATA_DIR / subdir / filename)

    for label in clean_paths:
        print(f"Class {label}: {len(clean_paths[label])}")

    return clean_paths


def build_slide_splits(clean_paths):
    # Class 4 (ambiguous) excluded from training
    mitotic = [(p, 1) for p in clean_paths["2"]]
    non_mitotic = [(p, 0) for p in clean_paths["1"] + clean_paths["3"] + clean_paths["7"]]

    print(f"Mitotic: {len(mitotic)}, Non-mitotic: {len(non_mitotic)}, Total: {len(mitotic) + len(non_mitotic)}")

    # Map uid -> slide using metadata
    df_annots = pd.read_csv(TRAIN_META_DATA_DIR / "Annotations.csv")
    uid_to_slide = dict(zip(df_annots["uid"], df_annots["slide"]))

    slide_groups = defaultdict(list)
    dropped = 0
    for path, label in mitotic + non_mitotic:
        uid = int(path.stem.split("_")[0])
        slide = uid_to_slide.get(uid)
        if slide is not None:
            slide_groups[slide].append((path, label))
        else:
            dropped += 1
    if dropped:
        print(f"Warning: {dropped} patches dropped (UID not found in Annotations.csv)")

    train_data, val_data = [], []

    for slide in TRAIN_SLIDES:
        train_data.extend(slide_groups[slide])
    for slide in VAL_SLIDES:
        val_data.extend(slide_groups[slide])

    for name, data in [("Train", train_data), ("Val", val_data)]:
        mit = sum(1 for _, l in data if l == 1)
        non = sum(1 for _, l in data if l == 0)
        print(f"{name:5s} | Mitotic: {mit:5d} | Non-mitotic: {non:5d} | Total: {mit+non:5d}")

    return train_data, val_data

def get_dataloaders(train_data, val_data):

    train_transform = get_train_transform()
    val_transform = get_eval_transform()

    train_dataset = MitosisDataset(train_data, transform=train_transform)
    val_dataset = MitosisDataset(val_data, transform=val_transform)

    # Weighted sampler to handle class imbalance
    train_labels = [label for _, label in train_data]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return train_loader, val_loader


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)

    return running_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):

    model.eval()
    val_loss = 0.0

    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()

            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().long().numpy())

    # Returns average val loss across all batches and average val accuracy across all validation examples.
    return val_loss / len(loader), correct / total, f1_score(all_labels, all_preds, pos_label=1)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    clean_paths = load_clean_paths()
    train_data, val_data = build_slide_splits(clean_paths)
    train_loader, val_loader = get_dataloaders(train_data, val_data)

    # Model
    model = build_model(pretrained=True)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    # Train
    best_val_f1 = 0.0

    for epoch in range(NUM_EPOCHS):

        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_f1)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Used Learning Rate: {current_lr} | Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "model.pth")
            print(f"  --> New best val F1: {best_val_f1:.3f} — model saved.")

    print(f"Training complete. Best val F1: {best_val_f1:.3f}")


if __name__ == "__main__":
    main()