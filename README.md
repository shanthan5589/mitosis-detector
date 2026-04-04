# Mitosis Detector — Two-Stage Pipeline

Binary detection of mitotic figures in histopathology whole-slide images of canine cutaneous mast cell tumors (CCMCT).

## Dataset

32 whole-slide DICOM images (21 training + 11 test) with 262,481 expert annotations across all 32 slides. This project uses the 21 training slides.

**Source:** [Training](https://www.kaggle.com/datasets/marcaubreville/mitosis-wsi-ccmct-training-set) | [Testing](https://www.kaggle.com/datasets/marcaubreville/mitosis-wsi-ccmct-test-set/data)

**Reference:** Bertram, C. A., Aubreville, M., Marzahl, C., Maier, A., & Klopfleisch, R. (2019). A large-scale dataset for mitotic figure assessment on whole slide images of canine cutaneous mast cell tumor. *Scientific Data*, 6(274), 1–9.

**Annotation classes:**

| Class ID | Name |
|----------|------|
| 1 | Granulocyte |
| 2 | Mitotic figure |
| 3 | Tumor cell |
| 4 | Other/ambiguous |
| 5 | Binucleated cell |
| 6 | Multinucleated cell |
| 7 | Mitotic figure lookalike |

Classes 5 and 6 were defined in the annotation schema but no cells of those classes were annotated in the training or testing slides.

---

## Why Two Stages?

The first version of this project used a single ResNet-18 classifier trained on 64×64 patches extracted from the slides. Despite trying many regularization techniques, architectures, and training strategies, the best validation F1 achieved was ~0.41. The root cause was domain shift — staining appearance varied significantly across slides, and a single classifier could not generalize across that gap.

The techniques attempted that did not work:
- Weighted Random Sampler
- pos_weight in BCEWithLogitsLoss
- Freezing all layers except the FC layer
- Freezing all layers except Layer-4 and FC
- Brightness/contrast augmentation
- Resizing input images
- Reducing learning rate / LR scheduling
- Modifying ResNet-18's conv1 and maxpool (breaks transfer learning)
- ResNet-50
- EfficientNet-B0

The core problem was that the single-stage model was trained on all extracted patches — it saw easy negatives (plain tissue, granulocytes) and never learned to distinguish the genuinely hard cases: cells that look like mitoses but aren't.

The two-stage approach addresses this directly:

**Stage 1 (YOLOv8s):** A detector tuned for high recall. Its job is to flag everything that might be a mitotic figure, at the cost of many false positives. Confidence threshold is deliberately set low (0.10).

**Stage 2 (EfficientNet-B2):** A classifier trained exclusively on the hard cases — YOLO's true positives (real mitoses it found), hard negatives (non-mitotic cells that fooled YOLO), and false negatives (mitoses YOLO missed). This forces the classifier to solve the genuinely difficult discrimination problem instead of the easy one.

---

## Project Structure

```
mitosis-detector/
├── README.md
├── requirements.txt
├── config.py                              # Centralized paths and hyperparameters
├── evaluate.py                            # Full pipeline evaluation (Stage 1 → Stage 2)
├── stage1_yolo/
│   ├── prepare_yolo_data.py              # Extract tiles from DICOM slides for YOLO
│   └── train_yolo.py                     # Train YOLOv8s detector
└── stage2_classifier/
    ├── prepare_stage2_data.py            # Build classifier training data from YOLO output
    └── train.py                          # Train EfficientNet-B2 classifier
```

---

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/shanthan5589/mitosis-detector.git
cd mitosis-detector
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install kaggle
```

**Kaggle API setup:** Go to kaggle.com → Your Profile → Settings → API → Create New Token. This downloads `kaggle.json`. Place it at:
- Linux/Mac: `~/.kaggle/kaggle.json`
- Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

### 3. Download the dataset

```bash
kaggle datasets download -d marcaubreville/mitosis-wsi-ccmct-training-set -p "data" --unzip
```

```bash
kaggle datasets download -d marcaubreville/mitosis-wsi-ccmct-test-set -p "data" --unzip
```

### 4. Configure paths

Edit `config.py` and set `TRAIN_DATA_DIR` and `TEST_DATA_DIR` to point to your local data. The dataset ships with a SQLite database (`MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite`) instead of CSVs. Extract the annotation tables first:

```python
import sqlite3, pandas as pd
from pathlib import Path

db_path = Path("path/to/MITOS_WSI_CCMCT_ODAEL_train_dcm.sqlite")
meta_dir = Path("path/to/training_data/meta_data")
meta_dir.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(db_path)
pd.read_sql("SELECT * FROM Slides", con).to_csv(meta_dir / "Slides.csv", index=False)
pd.read_sql("SELECT * FROM Annotations", con).to_csv(meta_dir / "Annotations.csv", index=False)
pd.read_sql("SELECT * FROM Annotations_coordinates", con).to_csv(meta_dir / "Annotations_coordinates.csv", index=False)
con.close()
```

### 5. Extract YOLO tiles from DICOM slides

```bash
python stage1_yolo/prepare_yolo_data.py
```

Outputs tile images and YOLO label files to `yolo_data/images/` and `yolo_data/labels/`. Slides are processed in parallel (one worker per slide). Note: reruns wipe and re-extract all tiles from scratch.

### 6. Train the YOLO detector

```bash
python stage1_yolo/train_yolo.py
```

Trains YOLOv8s for 50 epochs. Best weights saved to `models/yolo/weights/best.pt`.

### 7. Build Stage 2 training data

```bash
python stage2_classifier/prepare_stage2_data.py
```

Runs the trained YOLO model over all tiles. Extracts 96×96 crops and labels them as true positives, hard negatives, or false negatives. Writes CSV manifests to `stage2_data/`.

### 8. Train the Stage 2 classifier

```bash
python stage2_classifier/train.py
```

Trains EfficientNet-B2 for 30 epochs. Best checkpoint saved to `models/stage2_best.pth`.

### 9. Evaluate

```bash
python evaluate.py --split val
```

Runs the full pipeline (YOLO → EfficientNet) on the val slides end-to-end from raw DICOM and reports Precision, Recall, and F1.

```bash
python evaluate.py --split test
```

Requires `TEST_SLIDES` to be populated in `config.py` first.

> **Note:** `--split val` is optimistically biased — the same val slides were used to select the Stage 2 checkpoint. The `--split test` result on the held-out test set is the honest number.

---

## Data Pipeline

### DICOM whole-slide images

Each WSI is a tiled DICOM file. A single slide can be several gigabytes. The extraction pipeline in `prepare_yolo_data.py`:

1. Reads slide metadata (tile dimensions, total matrix size) from DICOM headers
2. Identifies which tiles contain at least one mitotic annotation (class 2 only)
3. Samples a small number of unannotated tiles as background examples
4. Streams through tiles using `iter_pixels`, decoding only target tiles
5. Saves tiles as images with YOLO-format bounding box label files

This avoids loading entire slides into memory and processes all slides in parallel.

### EDA findings (from the single-stage experiment)

1,000 patches per class were sampled to assess data quality. Two problems were found:

| Class | Black padding >10% | Plain (intensity >220) |
|-------|-------------------|------------------------|
| Granulocyte | 18.2% | 46.1% |
| Mitotic figure | 17.7% | 27.6% |
| Tumor cell | 19.4% | 31.1% |
| Ambiguous | 19.1% | 37.1% |
| Mitotic figure lookalike | 19.0% | 27.7% |

54% of patches were garbage and had to be filtered before training. It also surfaced class 7 (mitotic figure lookalikes) as a particularly relevant class — cells that visually resemble mitoses but are not, making them natural hard negatives for a classifier.

### Slide-level splits

Tiles were not split randomly. If tiles from the same slide appear in both train and val, the model memorizes slide-specific staining and gets rewarded for it — that is not generalization.

All data from a given slide goes entirely into one set. The counts below are from the original single-stage patch extraction and illustrate why the split was designed this way:

| Slide | Mitotic | Non-mitotic | Total | Split |
|-------|---------|-------------|-------|-------|
| 4 | 2,875 | 6,470 | 9,345 | Train |
| 7 | 948 | 3,485 | 4,433 | Val |
| 8 | 54 | 237 | 291 | Val |
| 12 | 847 | 3,351 | 4,198 | Train |
| 13 | 685 | 2,306 | 2,991 | Train |
| 14 | 1,183 | 1,708 | 2,891 | Val |
| 15 | 188 | 885 | 1,073 | Train |
| 17 | 371 | 1,774 | 2,145 | Train |
| 19 | 2,184 | 2,402 | 4,586 | Train |
| 21 | 2,547 | 3,672 | 6,219 | Train |
| 22 | 26 | 644 | 670 | Train |
| 23 | 0 | 1,328 | 1,328 | Val |
| 24 | 2 | 748 | 750 | Train |
| 25 | 7 | 1,077 | 1,084 | Train |
| 26 | 1 | 323 | 324 | Train |
| 28 | 0 | 606 | 606 | Train |
| 29 | 7 | 1,940 | 1,947 | Train |
| 32 | 6 | 1,412 | 1,418 | Train |
| 34 | 0 | 767 | 767 | Train |
| 35 | 19 | 1,351 | 1,370 | Train |
| 36 | 4 | 760 | 764 | Train |

| Split | Slides | Mitotic | Non-mitotic | Total |
|-------|--------|---------|-------------|-------|
| Train (17 slides) | 4, 12, 13, 15, 17, 19, 21, 22, 24, 25, 26, 28, 29, 32, 34, 35, 36 | 9,769 | 30,488 | 40,257 |
| Val (4 slides) | 7, 8, 14, 23 | 2,185 | 6,758 | 8,943 |

Slide 23 (zero mitotic cells) was deliberately placed in val. A model that has genuinely learned what mitosis looks like should produce very few false positives on it. If it doesn't, the model is matching noise.

---

## Model Architecture

### Stage 1 — YOLO Detector

| Setting | Value |
|---------|-------|
| Model | YOLOv8s |
| Input size | 640×640 |
| Epochs | 50 |
| Batch | 16 |
| Confidence threshold (inference) | 0.10 (intentionally low — maximize recall) |
| Augmentation | Horizontal/vertical flip, 90° rotation, HSV jitter. Mosaic, mixup, copy-paste disabled (distort cell morphology) |

### Stage 2 — EfficientNet-B2 Classifier

| Setting | Value |
|---------|-------|
| Model | EfficientNet-B2 (ImageNet pretrained) |
| Head | Linear(1408 → 1) replacing default classifier |
| Crop size | 96×96 (input to model: 224×224 after resize) |
| Epochs | 30 |
| Batch | 64 |
| Learning rate | 1e-4 (Adam) |
| LR scheduler | ReduceLROnPlateau, factor=0.5, patience=3 |
| Loss | BCEWithLogitsLoss with pos_weight (computed from class ratio) |
| Decision threshold | 0.50 |

### Stage 2 Training Data

| Crop type | Label | Description |
|-----------|-------|-------------|
| True Positive | 1 | Real mitosis that YOLO correctly detected |
| False Negative | 1 | Real mitosis that YOLO missed entirely |
| Hard Negative | 0 | Non-mitotic cell that fooled YOLO |

---

## Results

| | Precision | Recall | F1 | mAP50 |
|---|---|---|---|---|
| Stage 1 YOLO (val) | _(to be filled)_ | _(to be filled)_ | — | _(to be filled)_ |
| Full pipeline (val) | _(to be filled)_ | _(to be filled)_ | _(to be filled)_ | — |
| Full pipeline (test) | _(to be filled)_ | _(to be filled)_ | _(to be filled)_ | — |

---

## Key Learnings

- **A single classifier cannot solve this problem.** The domain gap between slides (staining variation) is too large for a patch classifier to generalize across. The max F1 achievable with a single ResNet-18/50 or EfficientNet-B0 was ~0.41 across all attempted training strategies.

- **Hard negative mining is the right fix.** Training Stage 2 only on cases that fooled YOLO — rather than random non-mitotic patches — directly targets the failure mode. The model is forced to learn the genuinely hard discrimination.

- **Two-stage recall-then-refine is a principled design.** Stage 1 is intentionally biased toward recall (conf threshold 0.10, no penalty for false positives). Stage 2 handles precision. Each stage has a single clear objective.

- **Slide-level splits are non-negotiable in medical imaging.** Random splits produce inflated, misleading metrics. Proper splits reveal the true generalization difficulty.

- **DICOM WSIs require careful streaming.** Loading even one slide fully into memory is impractical. `iter_pixels` allows tile-by-tile streaming — only decoding what's needed.

- **Parallelism across slides, not within.** Each slide is fully independent — processing them in parallel with `multiprocessing.Pool` gives near-linear speedup with cores. Within a single slide, streaming is inherently sequential.

- **Mosaic augmentation harms histology models.** YOLO's default mosaic stitches 4 random images together — this creates unnatural tissue boundaries that don't exist in real slides and actively degrades detection performance on WSI data.

- **Transfer learning assumptions break at small input sizes.** ResNet-18 pretrained on 224×224 ImageNet images loses most spatial information when fed 64×64 patches — feature maps collapse to 2×2 by layer4, making the pretrained weights mostly useless. This was a key reason to move to larger crops (96×96) in Stage 2.

- **Debugging experiments systematically is more valuable than running more experiments.** Every confirmed failure mode (domain shift, hard negatives, tile-boundary detection duplicates) led to a concrete architectural decision. Blind hyperparameter tuning did not.
