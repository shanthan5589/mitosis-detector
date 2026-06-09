# Mitosis Detector

**High-Precision Mitotic Figure Detection in Gigapixel Whole-Slide Pathology Images**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00b300.svg)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)

A robust two-stage deep learning pipeline that first maximizes recall with YOLOv8s, then refines predictions using an EfficientNet-B2 classifier trained on hard negatives. Achieves **F1 = 0.7738** on 11 fully held-out test slides with significant staining variation.

## Why This Matters

Mitotic figure counting is essential for tumor grading in histopathology, yet it remains a manual, time-consuming, and subjective task for pathologists. Whole-slide images (WSIs) are gigabytes in size and contain tens of thousands of cells. Staining variations across slides make generalization extremely difficult for conventional models.

This project solves the problem with a carefully designed two-stage approach that prioritizes both sensitivity and precision while rigorously preventing data leakage through slide-level splits.

The complete pipeline has been deployed as a REST API using FastAPI, containerized with Docker, and hosted on AWS EC2, exposing precise mitotic coordinate predictions through a live public endpoint for automated pathology workflows.

## Key Results

| Split | Precision | Recall | F1    |
|-------|-----------|--------|-------|
| Validation | 0.6981    | 0.8047 | 0.7476 |
| **Test (held-out)** | **0.7700**    | **0.7776** | **0.7738** |

The model generalizes across domain shifts where single-stage classifiers typically fail due to visually similar lookalikes (granulocytes, tumor cells, mitotic figure lookalikes).

---

## Features

- **Two-stage architecture**: High-recall YOLOv8s detector followed by precision-focused EfficientNet-B2 classifier trained exclusively on hard cases
- **Robust to domain shift**: Maintains high performance across slides with different staining protocols
- **Efficient large-image handling**: Streams gigabyte-scale DICOM whole-slide images tile-by-tile without loading entire slides into memory
- **Leakage-free evaluation**: Strict slide-level train/validation/test splits prevent the model from memorizing slide-specific artifacts
- **Hard negative mining**: Stage 2 is trained only on true mitoses and the most confusing false positives — dramatically improving the decision boundary
- **Reproducible end-to-end pipeline**: From raw DICOM to final mitotic coordinates with clear checkpoints and evaluation scripts

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Results](#results)
- [Key Learnings](#key-learnings)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Installation

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU recommended (training was done on T4 GPUs)
- Kaggle account (for dataset download)

### Step 1: Clone the repository
```bash
git clone https://github.com/shanthan5589/mitosis-detector.git
cd mitosis-detector
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
pip install kaggle
```

### Step 3: Set up Kaggle API credentials
1. Go to [kaggle.com](https://www.kaggle.com) → Your Profile → Settings → API → **Create New Token**
2. Place the downloaded `kaggle.json` file at:
   - Linux/macOS: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### Step 4: Download the datasets
```bash
# Training set (21 slides)
kaggle datasets download -d marcaubreville/mitosis-wsi-ccmct-training-set -p "data" --unzip

# Test set (11 slides)
kaggle datasets download -d marcaubreville/mitosis-wsi-ccmct-test-set -p "data" --unzip
```

> **Note**: Each whole-slide image is several gigabytes. Ensure you have sufficient disk space.

### Step 5: Configure paths
Edit `config.py` and set:
- `TRAIN_DATA_DIR` → path to your unzipped training folder
- `TEST_DATA_DIR` → path to your unzipped test folder (for final evaluation)

The scripts expect these metadata files inside each split's `meta_data/` folder:
- `Slides.csv`
- `Annotations.csv`
- `Annotations_coordinates.csv`

If your dataset provides SQLite files instead, generate the CSVs with:
```bash
python setup_data.py --split train
python setup_data.py --split test
```

## Quick Start

Run the steps **in order**. Each step depends on artifacts from the previous one.

```bash
# 1. Extract tiles and prepare YOLO training data
python stage1_yolo/prepare_yolo_data.py

# 2. Train YOLOv8s detector (high-recall stage)
python stage1_yolo/train_yolo.py

# 3. Generate hard-case training data for Stage 2
python stage2_classifier/prepare_stage2_data.py

# 4. Train EfficientNet-B2 classifier
python stage2_classifier/train.py

# 5. Evaluate full pipeline
python evaluate.py --split val
python evaluate.py --split test   # after configuring TEST_SLIDES in config.py
```

### Execution Checklist
Use this to verify successful runs:
- [ ] `yolo_data/data.yaml` is created
- [ ] `models/yolo/weights/best.pt` exists
- [ ] `stage2_data/train/train.csv` and `val/val.csv` are generated
- [ ] `models/stage2_best.pth` is saved
- [ ] `evaluate.py` prints Precision / Recall / F1 scores

## Project Structure
```
mitosis-detector/
├── README.md
├── requirements.txt
├── config.py
├── evaluate.py
├── setup_data.py
├── stage1_yolo/
│   ├── prepare_yolo_data.py
│   └── train_yolo.py
├── stage2_classifier/
│   ├── prepare_stage2_data.py
│   └── train.py
├── yolo_data/          # Generated: tiles + YOLO labels
├── stage2_data/        # Generated: hard-case crops + CSVs
├── models/             # Generated: trained weights
└── data/               # Your downloaded Kaggle datasets
```

## How It Works

### The Problem with Single-Stage Approaches
Initial experiments treated mitosis detection as simple binary patch classification (64×64 patches → ResNet-18 / EfficientNet-B0).

**Why it failed**:
- Models quickly overfit to slide-specific staining patterns
- Training loss decreased while validation loss diverged
- Best validation F1 remained stuck around **0.41**
- The model learned trivial negatives (plain tissue, granulocytes) instead of focusing on hard lookalikes

**Root cause**: Random patch-level splitting allowed information leakage. The model memorized staining characteristics of training slides rather than learning true mitotic morphology.

### The Solution: Two-Stage Pipeline

**Stage 1 — YOLOv8s Detector (High Recall)**
- Input: 640×640 tiles extracted from whole-slide DICOM images
- Confidence threshold deliberately set low (`0.10`) to minimize missed mitoses
- Accepts higher false positive rate in exchange for near-maximum sensitivity
- Mosaic / mixup augmentation disabled (they distort cell morphology)

**Stage 2 — EfficientNet-B2 Classifier (Precision Refinement)**
Trained **only** on three types of examples produced by Stage 1:
| Crop Type       | Label | Purpose |
|-----------------|-------|---------|
| True Positive   | 1     | Real mitoses correctly detected by YOLO |
| False Negative  | 1     | Real mitoses missed by YOLO |
| Hard Negative   | 0     | Non-mitotic cells that fooled YOLO |

This focuses model capacity exactly where it matters: distinguishing real mitotic figures from visually similar impostors.

**Why this works**:
- Stage 1 casts a wide net across the entire gigapixel slide
- Stage 2 specializes in the difficult decision boundary
- Result: strong generalization even when staining characteristics differ between training and test slides

## Results

### Performance on Held-Out Data
| Metric     | Validation | Test (11 slides) |
|------------|------------|------------------|
| Precision  | 0.6981     | 0.7700           |
| Recall     | 0.8047     | 0.7776           |
| **F1**     | 0.7476     | **0.7738**       |

### Training Curves
![YOLOv8s training curves](assets/yolo_curves.png)
![EfficientNet-B2 training curves](assets/stage2_curves.png)
![Single-stage overfitting pattern](assets/loss_curve.png)

## Key Learnings

- **Hard negative mining is critical** — Training on random negatives is insufficient when the real challenge is visually similar lookalikes.
- **Slide-level splits are non-negotiable** in medical imaging. Random splits produce misleadingly high metrics.
- **Two-stage recall-then-refine** is the right inductive bias for this task.
- **DICOM streaming** (`iter_pixels`) is essential — loading full WSIs into memory is impractical.
- **Mosaic augmentation hurts** histology performance by creating artificial boundaries that do not exist in real tissue.
- **Small input sizes weaken transfer learning** — 64×64 patches lose too much spatial information compared with 96×96 crops (Stage 2) or 640×640 tiles (Stage 1).

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. We deeply appreciate any help — whether it's reporting bugs, suggesting improvements, fixing documentation, or contributing code.

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ways to Help (Great for New Contributors)
- Improve documentation or add usage examples
- Add unit tests for data loading or evaluation
- Experiment with new backbones or augmentation strategies
- Optimize tile extraction or inference speed
- Report issues with clear reproduction steps

Please open an issue first if you're unsure where to start. We follow standard GitHub flow and are happy to guide new contributors.

## Acknowledgements

This project would not be possible without the outstanding dataset and annotations provided by:

**Bertram, C. A., Aubreville, M., Marzahl, C., Maier, A., & Klopfleisch, R. (2019).**  
A large-scale dataset for mitotic figure assessment on whole slide images of canine cutaneous mast cell tumor.  
*Scientific Data*, 6(274), 1–9. https://doi.org/10.1038/s41597-019-0290-4

Dataset access:
- [Training Set](https://www.kaggle.com/datasets/marcaubreville/mitosis-wsi-ccmct-training-set)
- [Test Set](https://www.kaggle.com/datasets/marcaubreville/mitosis-wsi-ccmct-test-set)

Special thanks to the pathologists who performed the exhaustive 262,481 expert annotations across 32 whole-slide images.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details (to be added).

---