# Model Card — EfficientNetB0 Brain Tumor MRI Classifier

A lightweight model card in the spirit of Mitchell et al. (2019), "Model Cards for Model Reporting".

## Model details

- **Developed by:** Raj Yadawad (portfolio / educational project).
- **Architecture:** EfficientNetB0 (ImageNet-pretrained backbone) + custom head
  (GlobalAveragePooling → Dropout → Dense(128, ReLU) → Dropout → Dense(4, softmax)).
- **Input:** RGB brain MRI image, resized to 224×224; normalization is in-graph.
- **Output:** probability distribution over 4 classes — `glioma`, `meningioma`, `notumor`,
  `pituitary`.
- **Training:** two-stage transfer learning (frozen head → fine-tune top 20 backbone layers),
  `sparse_categorical_crossentropy`, Adam, `EarlyStopping` on `val_loss`.
- **Version / framework:** TensorFlow 2.21 / Keras 3 (released checkpoint).
- **License:** MIT.

A from-scratch CNN is also released (`models/cnn_model.keras`) as an honest baseline.

## Intended use

- **Intended:** demonstrating an explainable, calibrated medical-imaging pipeline for education,
  portfolio review, and as a reference implementation of Grad-CAM / calibration / selective
  prediction in Keras 3.
- **Out of scope:** any real diagnostic or treatment decision. **This is not a medical device.**

## Training & evaluation data

- **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
  by Masoud Nickparvar (a merge of figshare + SARTAJ + Br35H).
- **Split:** official, class-balanced `Training/` (1,400/class) and `Testing/` (400/class). Validation
  is a seeded 20% split of `Training/`. `Testing/` is held out — never seen during training.

## Metrics (held-out test set, 1,600 images, TF 2.19 CPU)

| Metric            | EfficientNetB0 | CNN baseline |
|-------------------|----------------|--------------|
| Accuracy          | 85.3%          | 71.0%        |
| Macro F1          | 0.847          | 0.700        |
| ECE (10-bin)      | 0.034          | 0.130        |
| Errors            | 235 / 1600     | 464 / 1600   |

Most residual confusion is between *glioma* and *meningioma*. The model is well-calibrated and more
confident when correct, enabling a meaningful confidence-tiered escalation policy.

## Ethical considerations & limitations

- **Single dataset, no external validation** — performance on other scanners/populations is unknown.
- **Class balance is curated** (some classes augmented to balance) — real clinical prevalence differs.
- **False negatives carry clinical cost** — a tumour read as *No Tumor* is the highest-severity error;
  the project's escalation policy is designed to route uncertain cases to humans precisely for this
  reason.
- **No segmentation / localisation** — Grad-CAM gives coarse attention, not a clinical delineation.
- **Not validated, not regulated** — must not be used for diagnosis or treatment.

## Caveats

The released checkpoints are CPU-trained with modest epochs; results are a portfolio demonstration,
not a state-of-the-art benchmark. See [`reproducibility.md`](reproducibility.md) for the full
environment and how to regenerate every number and figure.
