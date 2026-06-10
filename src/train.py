"""
train.py
========
Train both models end-to-end from the command line::

    python -m src.train                  # full training
    python -m src.train --cnn-epochs 3 --tl-epochs 2 --ft-epochs 1   # quick smoke run

This mirrors the notebook's Phase 2 and bakes in the audit's H1/H2 fix:

* Each model is trained with ``EarlyStopping(restore_best_weights=True)`` so the
  best-val_loss weights are left in memory.
* Stage-2 fine-tuning uses a **fresh** EarlyStopping (never the Stage-1 callback),
  so it cannot accidentally leave Stage-1 weights on disk.
* After training, each model is **saved and then reloaded and re-evaluated**, so the
  released file is provably the model we report on.
"""

from __future__ import annotations

import argparse
import os

from . import config
from .data_loader import build_datasets
from .models import build_cnn, build_transfer
from .utils import get_logger, set_seed

logger = get_logger(__name__)


def _callbacks(min_lr: float):
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=2, min_lr=min_lr),
    ]


def train_cnn(train_ds, val_ds, epochs: int):
    cnn = build_cnn()
    cnn.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=_callbacks(1e-5))
    cnn.save(config.CNN_PATH)
    logger.info("Saved CNN -> %s", config.CNN_PATH)
    return cnn


def train_efficientnet(train_ds, val_ds, tl_epochs: int, ft_epochs: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    effnet, base = build_transfer()

    logger.info("Stage 1: training head (base frozen)...")
    effnet.fit(train_ds, validation_data=val_ds, epochs=tl_epochs, callbacks=_callbacks(1e-6))

    logger.info("Stage 2: fine-tuning top base layers...")
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    for layer in base.layers[-20:]:
        if isinstance(layer, layers.BatchNormalization):   # keep BN stats frozen
            layer.trainable = False
    effnet.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # Fresh callbacks for Stage 2 (H1 fix).
    effnet.fit(train_ds, validation_data=val_ds, epochs=ft_epochs, callbacks=_callbacks(1e-6))

    effnet.save(config.EFFNET_PATH)
    logger.info("Saved EfficientNet -> %s", config.EFFNET_PATH)
    return effnet


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CNN and EfficientNet models.")
    parser.add_argument("--cnn-epochs", type=int, default=config.CNN_EPOCHS)
    parser.add_argument("--tl-epochs", type=int, default=config.TL_EPOCHS)
    parser.add_argument("--ft-epochs", type=int, default=config.FT_EPOCHS)
    parser.add_argument("--skip-cnn", action="store_true")
    args = parser.parse_args()

    set_seed()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    train_ds, val_ds, test_ds, _ = build_datasets()

    if not args.skip_cnn:
        train_cnn(train_ds, val_ds, args.cnn_epochs)
    train_efficientnet(train_ds, val_ds, args.tl_epochs, args.ft_epochs)

    # Verify: reload from disk and evaluate, so reported == released.
    from .utils import load_model
    for label, path in [("CNN", config.CNN_PATH), ("EfficientNet", config.EFFNET_PATH)]:
        if os.path.exists(path):
            _, acc = load_model(path).evaluate(test_ds, verbose=0)
            logger.info("Reloaded %s | test accuracy = %.4f", label, acc)


if __name__ == "__main__":
    main()
