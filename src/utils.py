"""
utils.py
========
Cross-cutting helpers: reproducibility, logging, and model loading.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

from . import config


def set_seed(seed: int = config.SEED) -> None:
    """Seed Python, NumPy and TensorFlow so runs are reproducible.

    Importing TensorFlow is deferred to keep ``set_seed`` cheap for callers
    (e.g. tests) that only need the Python/NumPy RNGs seeded.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:  # pragma: no cover - TF always present at runtime
        pass


def get_logger(name: str = "brain_mri") -> logging.Logger:
    """Return a module logger with a single stream handler (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def load_model(path: str):
    """Load a saved ``.keras`` model, with a clear error if it is missing.

    Kept as a one-liner wrapper so callers don't import TensorFlow directly and
    so the "train first" hint lives in one place.
    """
    import tensorflow as tf

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Train the models first (see src/train.py or the notebook), or download "
            "the released checkpoints into models/."
        )
    return tf.keras.models.load_model(path)
