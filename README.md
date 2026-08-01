# 🩺 Pneumonia Detection using Transfer Learning with ResNet50V2

Deep learning for binary classification of chest X-ray images into **PNEUMONIA** and **NORMAL**, using transfer learning on a pretrained **ResNet50V2** backbone.

Two experimental stages are explored in the notebooks in this repository:

| Notebook | Approach | Test Accuracy | AUC-ROC |
|---|---|---|---|
| [Version 1](pneumonia-detection-tl-resnet50v2-version1.ipynb) | Fine-tuned last 30 layers of ResNet50V2 | 81% | **93%** |
| [Version 2](pneumonia-detection-tl-resnet50v2-version2.ipynb) | Frozen backbone (feature extraction only) | 84% | **96%** |

> **Key finding:** Freezing the pretrained backbone (V2) generalized better than fine-tuning the last 30 layers (V1), improving AUC from **93% → 96%** while reducing overfitting.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Approach](#approach)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Model Architecture](#model-architecture)
- [Training Setup](#training-setup)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Challenges & Limitations](#challenges--limitations)
- [Conclusions](#conclusions)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## Project Overview

Pneumonia is one of the leading causes of death worldwide, especially among children and the elderly. Early diagnosis from chest X-ray images can significantly improve treatment outcomes.

This project applies transfer learning by leveraging a **ResNet50V2** model pretrained on ImageNet to classify chest X-ray images into two categories:

- **PNEUMONIA**
- **NORMAL**

The project also investigates how **class imbalance** affects model performance and evaluates different transfer learning strategies (fine-tuning vs. feature extraction) to improve generalization.

---

## Objectives

- Load and preprocess chest X-ray images.
- Build a transfer learning model using ResNet50V2.
- Compare two strategies:
  - **V1:** Fine-tuning the last 30 layers of the base model with a custom classification head.
  - **V2:** Using a fully frozen backbone (feature extraction) with a custom classification head.
- Reduce the overfitting observed in the fine-tuned model.
- Evaluate performance using accuracy, recall, confusion matrix, and **AUC-ROC**.

---

## Dataset

**Source:** [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset (Kaggle, `paultimothymooney`).

**Classes:** `NORMAL`, `PNEUMONIA`

| Split | Pneumonia | Normal |
|---|---|---|
| Training | 3,875 | 1,341 |
| Testing | 390 | 234 |

- **Image format:** JPEG
- **Input size:** 224 × 224
- **Note:** The dataset is imbalanced (roughly 3:1 pneumonia-to-normal ratio in training).

> The original validation split contained only 16 images, which is insufficient for reliable evaluation. Both notebooks therefore carve out **15% of the training data as a stratified validation set** (preserving class distribution).

---

## Approach

Both notebooks share the same pipeline but differ in how the pretrained backbone is handled:

- **Version 1 (Fine-tuning):** All convolutional layers are frozen *except* the last 30, which are trained together with the custom classification head.
- **Version 2 (Feature extraction):** The entire ResNet50V2 backbone is frozen (`base_model.trainable = False`); only the classification head is trained.

This isolates the effect of backbone fine-tuning on generalization for small medical datasets.

---

## Preprocessing Pipeline

1. **Read** each image in grayscale (`cv2.IMREAD_GRAYSCALE`).
2. **Resize** to 224 × 224.
3. **Convert** to NumPy arrays and reshape to `(n, 224, 224, 1)`.
4. **Normalize** pixel values to `[0, 1]` (divide by 255).
5. **Convert grayscale → RGB** by repeating the single channel (`np.repeat(..., 3, axis=-1)`), as required by ResNet50V2.
6. Apply ResNet50V2-specific `preprocess_input`.
7. **Split** into train/validation using stratified sampling (15% validation).

---

## Model Architecture

```
ResNet50V2 (ImageNet, include_top=False, input_shape=(224, 224, 3))
        │
        ▼
GlobalAveragePooling2D
        │
        ▼
Dense(512, activation="relu")
        │
        ▼
Dropout(0.5)
        │
        ▼
Dense(128, activation="relu")
        │
        ▼
Dropout(0.3)
        │
        ▼
Dense(1, activation="sigmoid")
```

- **Output:** single sigmoid unit → binary classification.
- **Head regularization:** dropout layers (0.5, 0.3) to reduce overfitting.

---

## Training Setup

- **Optimizer:** Adam
- **Loss:** Binary cross-entropy
- **Metrics:** Accuracy
- **Batch size:** 64
- **Epochs:** 10 (V1) / 20 (V2)
- **Class weights:** `compute_class_weight("balanced")` applied during `fit()` to counter class imbalance.
- **Distribution strategy:** `MirroredStrategy` (multi-GPU, Kaggle T4 ×2).

### Callbacks

| Callback | Setting |
|---|---|
| `EarlyStopping` | monitor `val_loss`, patience 5, restore best weights |
| `ReduceLROnPlateau` | factor 0.2, patience 2, min LR 1e-6 |
| `ModelCheckpoint` | save best `best_resnet50v2.keras` by `val_accuracy` |

### Data Augmentation

Applied on-the-fly to improve generalization:

- Random horizontal flip
- Random rotation (0.1)
- Random zoom (0.1)
- Random contrast (0.1)

---

## Evaluation Metrics

- **Test accuracy & loss** (`model.evaluate`).
- **AUC-ROC** and ROC curve plot (`roc_auc_score`, `roc_curve`).
- **Confusion matrix** and `classification_report` (precision / recall / F1 per class).

---

## Results

### Version 1 — Fine-tuned last 30 layers (AUC: 93%)

- **Test accuracy:** 81%
- **AUC-ROC:** 0.93
- **Pneumonia recall:** 99% (386 / 390)
- **Normal recall:** 51% (114 / 234)
- **False positives:** 120 normal images misclassified as pneumonia

### Version 2 — Frozen backbone / feature extraction (AUC: 96%)

- **Test accuracy:** 84%
- **AUC-ROC:** 0.96
- **Pneumonia recall:** 98% (384 / 390)
- **Normal recall:** 65% (83 / 234)
- **False positives:** 151 normal images misclassified as pneumonia

### Summary table

| Metric | V1 (Fine-tuned) | V2 (Feature extraction) |
|---|---|---|
| Test accuracy | 81% | 84% |
| AUC-ROC | **93%** | **96%** |
| Pneumonia recall | 99% | 98% |
| Normal recall | 51% | 65% |

Both models are biased toward predicting pneumonia, but the frozen-backbone model (V2) yields better generalization and a higher AUC, confirming that fine-tuning on this small, imbalanced dataset tends to overfit.

---

## Challenges & Limitations

- **Class imbalance:** ~3:1 pneumonia-to-normal ratio.
- **Overfitting:** both models reach high training/validation accuracy but struggle on unseen data.
- **High false-positive rate:** many NORMAL X-rays are predicted as pneumonia.
- **Class weighting + augmentation** were evaluated but did not significantly improve test performance.
- **Small original validation set:** had to be replaced with a stratified split of training data.

---

## Conclusions

This project demonstrates the effectiveness of transfer learning for medical image classification.

- Fine-tuning the backbone (V1) achieved 93% AUC but suffered from limited generalization.
- Freezing the backbone (V2) achieved **96% AUC** with better generalization, making it the stronger approach for this task.
- Future improvements include backbone fine-tuning with stronger regularization, advanced imbalance-handling techniques, and threshold tuning to reduce false positives.

---

## Requirements

- Python 3
- TensorFlow / Keras
- NumPy, Pandas
- scikit-learn
- Matplotlib, Seaborn
- OpenCV (`cv2`)
- `kagglehub` (for dataset download)

---

## How to Run

1. Clone/download this repository and open either notebook:
   - `pneumonia-detection-tl-resnet50v2-version1.ipynb` (fine-tuning)
   - `pneumonia-detection-tl-resnet50v2-version2.ipynb` (feature extraction)
2. The dataset is downloaded automatically via `kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")`.
3. Install dependencies: `pip install tensorflow scikit-learn matplotlib seaborn opencv-python kagglehub`
4. Run all cells. Training uses `MirroredStrategy` (GPU recommended).
5. The best model is saved to `best_resnet50v2.keras` and the ROC curve to `roc_curve.png`.

> Both notebooks are designed to run on **Kaggle** with a GPU environment (T4 ×2), but will also run on a single GPU/CPU with longer training time.
