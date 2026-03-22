"""
Evaluation script: Loads a trained model and evaluates on the test set
with precision, recall, F1, and confusion matrix.

Usage:
    python evaluate.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix

from config import BASE_DIR, META_DATA_DIR, CLEAN_PATHS_FILE
from train import MitosisDataset, load_clean_paths, build_slide_splits, IMAGE_SIZE, BATCH_SIZE, TEST_SLIDES


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    clean_paths = load_clean_paths()
    _, _, test_data = build_slide_splits(clean_paths)

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = MitosisDataset(test_data, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = models.resnet18()
    model.fc = nn.Linear(512, 1)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # Evaluate
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=["Non-mitotic", "Mitotic"]))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
