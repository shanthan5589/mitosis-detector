import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from torchvision.ops import nms

from config import MODELS_DIR, YOLO_CONF_THRESH, STAGE2_CROP_SIZE, STAGE2_IMG_SIZE, STAGE2_CONF_THRESH
from stage2_classifier.prepare_stage2_data import extract_crop

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("pip install ultralytics")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

stage2_transform = transforms.Compose([
    transforms.Resize((STAGE2_IMG_SIZE, STAGE2_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load models once at startup
yolo_model = YOLO(str(MODELS_DIR / "yolo" / "weights" / "best.pt"))
yolo_model.to(str(device))

stage2_model = models.efficientnet_b2(weights=None)
in_features = stage2_model.classifier[1].in_features
stage2_model.classifier[1] = nn.Linear(in_features, 1)
stage2_model.load_state_dict(torch.load(str(MODELS_DIR / "stage2_best.pth"), map_location=device, weights_only=False))
stage2_model.to(device).eval()


def predict_tile(img_bgr: np.ndarray) -> dict:
    # Stage 1: YOLO
    results = yolo_model(img_bgr, conf=YOLO_CONF_THRESH, verbose=False)
    candidate_boxes = []
    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            candidate_boxes.append(box[:4])

    if not candidate_boxes:
        return {"mitotic_count": 0, "detections": []}

    # Stage 2: EfficientNet
    crops = []
    for box in candidate_boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        crops.append(extract_crop(img_bgr, cx, cy, STAGE2_CROP_SIZE))

    tensors = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tensors.append(stage2_transform(pil))

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        scores = torch.sigmoid(stage2_model(batch)).float().cpu().squeeze(1).tolist()

    # Filter by threshold
    passing = [(box, s) for box, s in zip(candidate_boxes, scores) if s >= STAGE2_CONF_THRESH]

    if not passing:
        return {"mitotic_count": 0, "detections": []}

    detections = []
    for box, score in passing:
        detections.append({
            "x": float((box[0] + box[2]) / 2),
            "y": float((box[1] + box[3]) / 2),
            "confidence": round(score, 4)
        })

    return {"mitotic_count": len(detections), "detections": detections}