"""
config.py
=========
Single source of truth for paths and hyperparameters across the Brain Tumor MRI
project. Every other module (and the notebook's logic) reads its constants from
here, so behaviour changes in exactly one place.

Paths are environment-overridable, which keeps the code portable across machines
and makes it runnable in CI without touching the source:

    BRAIN_MRI_DATA_DIR    -> root holding Training/ and Testing/ (default: <root>/Dataset)
    BRAIN_MRI_MODELS_DIR  -> where .keras checkpoints live      (default: <root>/models)
    BRAIN_MRI_RESULTS_DIR -> where metrics/figures are written  (default: <root>/results)
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
# .../Brain Tumor MRI Dataset/src/config.py  ->  .../Brain Tumor MRI Dataset
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_path(var: str, default: str) -> str:
    """Return an absolute path from an env var, falling back to ``default``."""
    return os.path.abspath(os.environ.get(var, default))


DATASET_DIR: str = _env_path("BRAIN_MRI_DATA_DIR", os.path.join(PROJECT_ROOT, "Dataset"))
TRAIN_DIR: str = os.path.join(DATASET_DIR, "Training")
TEST_DIR: str = os.path.join(DATASET_DIR, "Testing")

MODELS_DIR: str = _env_path("BRAIN_MRI_MODELS_DIR", os.path.join(PROJECT_ROOT, "models"))
RESULTS_DIR: str = _env_path("BRAIN_MRI_RESULTS_DIR", os.path.join(PROJECT_ROOT, "results"))
FIGURES_DIR: str = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR: str = os.path.join(RESULTS_DIR, "metrics")

CNN_PATH: str = os.path.join(MODELS_DIR, "cnn_model.keras")
EFFNET_PATH: str = os.path.join(MODELS_DIR, "efficientnet_model.keras")

# ---------------------------------------------------------------------------
# Data / model settings
# ---------------------------------------------------------------------------
IMAGE_SIZE: int = 224
BATCH_SIZE: int = 32
NUM_CLASSES: int = 4
VAL_SPLIT: float = 0.2
SEED: int = 42

# Folder names are the canonical class order (alphabetical, as Keras reads them).
CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
# Human-readable labels for plots and reports.
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

# Training length (EarlyStopping may stop sooner). Override via train.py CLI.
CNN_EPOCHS: int = 15
TL_EPOCHS: int = 10
FT_EPOCHS: int = 5

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
