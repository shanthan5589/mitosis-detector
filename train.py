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
from collections import defaultdict, Counter

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

    if not train_data or not val_data:
        raise ValueError("Train or val split is empty — check slide IDs and Annotations.csv")

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
    # train_labels = [label for _, label in train_data]
    # class_counts = [train_labels.count(0), train_labels.count(1)]
    # weights = [1.0 / class_counts[label] for label in train_labels]
    # sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(weights))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
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

    all_labels = []
    all_preds = []

    correct = 0
    total = 0

    best_f1 = 0.0
    best_thresh = 0.0

    all_probs = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze(1)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            probs = torch.sigmoid(outputs)

            all_probs.extend(probs)           # Here probs is a tensor, but all_probs is still a list. A list doesn't become a tensor if it *contains tensors. It's still a list. 

            preds = (probs > 0.5).cpu().long()
            all_preds.extend(preds.numpy())

            correct += (preds == labels.cpu().long()).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().long().numpy())

        avg_batch_loss = val_loss / len(loader)  # average val loss across all batches.
        val_accuracy = correct / total           # average val accuracy across all validation examples.
        calc_f1 = f1_score(all_labels, all_preds, pos_label=1)  # f1 for thresh = 0.5 (fixed)

        all_probs = torch.stack(all_probs)       # Since all_probs is a list we convert to tensor.

        for thresh in [i/100 for i in range(5, 96, 2)]:
            all_preds = (all_probs > thresh).cpu().long().numpy()
            current_f1 = f1_score(all_labels, all_preds, pos_label=1)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_thresh = thresh

    # val_accuracy uses hardcoded 0.5 as threshold in its calculations.
    # best_f1 is calculated on best found thresh value. (not a hardcoded 0.5 thresh)
    return avg_batch_loss , val_accuracy, calc_f1, best_f1, best_thresh


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

    labels = []
    for _, label in train_data:
        labels.append(int(label))
    
    counter = Counter(labels)

    mitotic = counter.get(1,0)
    non_mitotic = counter.get(0,0)

    if mitotic == 0:
        raise ValueError("No positive samples found!")
    
    print(f"pos_weight: {non_mitotic / mitotic:.4f}")

    pos_weight = torch.tensor([non_mitotic / mitotic]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Train
    best_val_f1 = 0.0

    for epoch in range(NUM_EPOCHS):

        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, curr_f1, val_f1, val_thresh = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Used Learning Rate: {current_lr} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | curr_F1: {curr_f1} | best_F1: {val_f1:.3f} | Thresh: {val_thresh}")

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            torch.save({
                "model_state": model.state_dict(),
                "best_thresh": val_thresh
            },  "model.pth")

            print(f"  --> New best val F1: {best_val_f1:.3f} — model saved.")

    print(f"Training complete. Best val F1: {best_val_f1:.3f}")


if __name__ == "__main__":
    main()