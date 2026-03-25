"""
Evaluation script: Loads a trained model and evaluates on the selected split
with precision, recall, F1, and confusion matrix.

Usage:
    python evaluate.py --split val    # evaluate on validation set
    python evaluate.py --split test   # evaluate on test set
"""

import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path

from config import TEST_DATA_DIR
from train import MitosisDataset, load_clean_paths, build_slide_splits
from model_utils import build_model, get_eval_transform, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY

def get_test_data():
    if not TEST_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Test data not found at {TEST_DATA_DIR}. "
            "Run setup_data.py --split test and extract_patches.py --split test first."
        )

    mitotic_dir = TEST_DATA_DIR / "mitotic"
    non_mitotic_dir = TEST_DATA_DIR / "non_mitotic"

    if not mitotic_dir.exists() or not non_mitotic_dir.exists():
        raise FileNotFoundError(
            f"mitotic/ or non_mitotic/ folders not found in {TEST_DATA_DIR}. "
            "Run extract_patches.py --split test first."
        )

    data = []
    for path in mitotic_dir.iterdir():
        if path.suffix == ".png":
            data.append((path, 1))
    for path in non_mitotic_dir.iterdir():
        if path.suffix == ".png":
            # Exclude class 4 (ambiguous) — no ground truth, consistent with training
            if path.stem.split("_")[-1] != "4":
                data.append((path, 0))

    if not data:
        raise RuntimeError(f"No patches found in {TEST_DATA_DIR}. Run extract_patches.py --split test first.")

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], required=True,
                        help="Which split to evaluate on: val or test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data based on split
    if args.split == "val":
        clean_paths = load_clean_paths()
        _, data = build_slide_splits(clean_paths)
    else:
        data = get_test_data()

    transform = get_eval_transform()

    dataset = MitosisDataset(data, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Load model
    model_path = Path("model.pth")
    if not model_path.exists():
        raise FileNotFoundError(
            "model.pth not found in current directory. "
            "Run train.py from this directory first."
        )
    
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze(1)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    split_name = "Validation" if args.split == "val" else "Test"
    print(f"\n{'='*40}")
    print(f"  {split_name} Set Results")
    print(f"{'='*40}")
    print(classification_report(all_labels, all_preds, target_names=["Non-mitotic", "Mitotic"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()