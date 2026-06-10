"""
data_loader.py
==============
Build the ``tf.data`` pipelines used for training and evaluation, mirroring the
notebook's preprocessing.

Split strategy (leakage-free):
* **Train / validation** are carved out of ``Training/`` via a single seeded
  ``validation_split`` — both models see the same split.
* **Test** is the untouched ``Testing/`` folder, kept **unshuffled** so predictions
  stay aligned with their ground-truth labels.
"""

from __future__ import annotations

import os

import tensorflow as tf

from . import config
from .utils import get_logger

logger = get_logger(__name__)

Dataset = tf.data.Dataset


def count_images_per_class(directory: str) -> dict[str, int]:
    """Count images per class **without** decoding pixels — fast enough for EDA."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    counts: dict[str, int] = {}
    for name in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, name)
        if not os.path.isdir(class_dir):
            continue
        counts[name] = sum(
            1 for f in os.listdir(class_dir)
            if f.lower().endswith(config.VALID_IMAGE_EXTS)
        )
    return counts


def build_datasets(
    train_dir: str = config.TRAIN_DIR,
    test_dir: str = config.TEST_DIR,
    image_size: int = config.IMAGE_SIZE,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    seed: int = config.SEED,
    prefetch: bool = True,
) -> tuple[Dataset, Dataset, Dataset, list[str]]:
    """Return ``(train_ds, val_ds, test_ds, class_names)``.

    Images are decoded and resized to ``image_size`` but **not** normalized — each
    model normalizes internally (see :mod:`src.preprocessing`).
    """
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Testing directory not found: {test_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="training", seed=seed,
        image_size=(image_size, image_size), batch_size=batch_size, label_mode="int")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=val_split, subset="validation", seed=seed,
        image_size=(image_size, image_size), batch_size=batch_size, label_mode="int")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=(image_size, image_size), batch_size=batch_size,
        label_mode="int", shuffle=False)   # unshuffled: labels align with predictions

    class_names = train_ds.class_names   # read before prefetch wraps the dataset
    logger.info("Class names: %s", class_names)

    if prefetch:
        autotune = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(autotune)
        val_ds = val_ds.prefetch(autotune)
        test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names
