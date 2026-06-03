# 🧠 Brain Tumor MRI Classification using Deep Learning and Explainable AI

An end-to-end medical imaging project that classifies brain MRI scans into four categories —
**glioma**, **meningioma**, **no tumor**, and **pituitary** — and goes beyond raw accuracy to
deliver **explainability (Grad-CAM)**, **reliability/calibration analysis**, and a simulated
**clinical decision-support system (CDSS)**.

The entire pipeline lives in a single, reproducible notebook:
[`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb).

---

## Overview

Convolutional neural networks can classify brain tumors from MRI with high accuracy, but in a
clinical setting accuracy alone is not enough — a model also needs to **explain** its
predictions and **know when it is uncertain**. This project builds a from-scratch CNN baseline,
upgrades to **EfficientNetB0 transfer learning**, and then layers on the trust-and-safety
analysis a real diagnostic-support tool would require: Grad-CAM visual explanations,
confidence calibration, error analysis, and a confidence-tiered escalation workflow for
human-in-the-loop review.

---

## Features

- **Data exploration** — class distribution, image-dimension/quality checks, sample grids.
- **CNN model** — a from-scratch convolutional baseline for an honest reference point.
- **EfficientNet transfer learning** — EfficientNetB0 pretrained on ImageNet, two-stage
  training (frozen head → fine-tuning the top layers).
- **Explainable AI (Grad-CAM)** — per-class heatmaps showing where the model "looks,"
  plus a tightness/attention assessment of those explanations.
- **Clinical decision support simulation** — risk stratification by confidence tier,
  auto-generated diagnostic reports, an escalation policy, and a clinical confidence dashboard.
- **Error analysis** — most/least-confident mistakes, class-confusion study, and the
  clinical cost of different error types (e.g. false negatives).
- **Reliability analysis** — confidence distributions, **Expected Calibration Error (ECE)**,
  and a selective-prediction (risk–coverage) study.

---

## Dataset

This project uses the **[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)**
from Kaggle (4 classes, ~7,200 images, with an official train/test split).

The dataset is **not committed** to this repository (≈178 MB). To run the notebook, download
it from Kaggle and place it so the structure is:

```
Dataset/
├── Training/   (glioma · meningioma · notumor · pituitary)
└── Testing/    (glioma · meningioma · notumor · pituitary)
```

| Split    | Images / class | Classes                                | Total |
|----------|----------------|----------------------------------------|-------|
| Training | 1,400          | glioma, meningioma, notumor, pituitary | 5,600 |
| Testing  | 400            | glioma, meningioma, notumor, pituitary | 1,600 |

---

## Project Workflow

```
Dataset → Preprocessing → Training → Evaluation → Explainability → Clinical Reporting
```

1. **Preprocessing** — load, resize to 224×224, normalize, stratified train/validation split,
   on-the-fly augmentation (rotation, flip, zoom).
2. **Training** — CNN baseline, then EfficientNetB0 transfer learning with fine-tuning.
3. **Evaluation** — accuracy, precision, recall, F1, confusion matrices, model comparison.
4. **Explainability** — Grad-CAM overlays and an explainability assessment.
5. **Clinical Reporting** — risk tiers, diagnostic reports, escalation policy, CDSS dashboard.

---

## Results

Evaluated on the held-out test set (1,600 images):

| Model            | Accuracy  | Macro Precision | Macro Recall | Macro F1 |
|------------------|-----------|-----------------|--------------|----------|
| CNN (baseline)   | 71.2%     | 0.725           | 0.712        | 0.703    |
| **EfficientNetB0** | **85.0%** | **0.855**       | **0.850**    | **0.844** |

**Key findings**
- Transfer learning lifts test accuracy by **~14 points** over the from-scratch CNN.
- The best model is well-calibrated and **more confident when correct** (mean test
  confidence ≈ **89.4%**), enabling a meaningful confidence-tier escalation policy.
- **Grad-CAM** confirms the model attends to the tumor region for confident, correct
  predictions — visual evidence that supports clinical trust.
- Error analysis treats mistakes by clinical cost (a tumor read as "No Tumor" is a
  high-severity false negative), motivating the human-in-the-loop escalation workflow.

---

## Technologies Used

- **Python**
- **TensorFlow / Keras** (CNN + EfficientNetB0 transfer learning, Grad-CAM)
- **NumPy**, **Pandas**
- **Matplotlib**
- **Scikit-learn** (metrics, calibration, classification reports)
- **Jupyter Notebook**

---

## How to Run

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the Kaggle dataset into ./Dataset (see Dataset section above)

# 4. Launch the notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

Pre-trained weights are included under `models/` (`cnn_model.keras`,
`efficientnet_model.keras`), so you can load the models for inference without retraining.

---

## Future Improvements

- **Calibration methods** — temperature scaling / Platt scaling to further reduce ECE.
- **Stronger backbones & ensembling** — EfficientNet-V2 / ConvNeXt and multi-model ensembles.
- **Cross-dataset validation** — test generalization on independent MRI sources and scanners.
- **Segmentation** — localize tumors (not just classify) for richer clinical output.
- **Deployment** — wrap the model in an inference API / web app with the CDSS report generator.
- **Clinical validation** — prospective evaluation with radiologist review before any real use.

---

## Author

**Raj Yadawad**

---

> ⚠️ **Disclaimer:** This project is for research and educational purposes only. It is **not**
> a medical device and must not be used for actual diagnosis or treatment decisions.
