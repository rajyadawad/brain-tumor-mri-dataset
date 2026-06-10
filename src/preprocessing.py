"""
preprocessing.py
================
On-the-fly data augmentation, mirroring the notebook.

Normalization is intentionally NOT done here: it lives *inside* each model
(a ``Rescaling`` layer for the CNN, EfficientNet's built-in normalization for the
transfer model). Keeping it in-graph means a saved model is self-contained and
cannot be fed wrongly-scaled inputs at inference time.
"""

from __future__ import annotations

from tensorflow.keras import Sequential, layers

from . import config


def make_augmentation(seed: int = config.SEED) -> Sequential:
    """Return a fresh augmentation block (flip, small rotation, small zoom).

    A factory (rather than a shared instance) lets the same augmentation be
    embedded in multiple models safely. Augmentation is active only when a model
    is called with ``training=True``.
    """
    return Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.05, seed=seed),   # ~ +/- 18 degrees
            layers.RandomZoom(0.10, seed=seed),
        ],
        name="data_augmentation",
    )
