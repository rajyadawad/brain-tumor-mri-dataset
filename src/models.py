"""
models.py
=========
Model definitions, mirroring the notebook:

* ``build_cnn``      — a from-scratch convolutional baseline (honest reference point).
* ``build_transfer`` — EfficientNetB0 transfer learning (the production model).

Both embed augmentation and normalization in-graph, and name their final conv
layer ``top_conv`` so Grad-CAM can target it.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from . import config
from .preprocessing import make_augmentation


def build_cnn(
    image_size: int = config.IMAGE_SIZE,
    num_classes: int = config.NUM_CLASSES,
) -> tf.keras.Model:
    """From-scratch CNN baseline.

    Four ``Conv -> BatchNorm -> ReLU -> MaxPool`` blocks, global average pooling,
    and a dropout-regularised dense head. The last conv layer is named ``top_conv``
    for Grad-CAM.
    """
    model = models.Sequential(
        [
            layers.Input((image_size, image_size, 3)),
            make_augmentation(),                 # augment during training only
            layers.Rescaling(1.0 / 255),         # normalize [0,255] -> [0,1]

            layers.Conv2D(32, 3, padding="same", use_bias=False),
            layers.BatchNormalization(), layers.Activation("relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, 3, padding="same", use_bias=False),
            layers.BatchNormalization(), layers.Activation("relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, padding="same", use_bias=False),
            layers.BatchNormalization(), layers.Activation("relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, 3, padding="same", use_bias=False, name="top_conv"),
            layers.BatchNormalization(), layers.Activation("relu"),
            layers.MaxPooling2D(),

            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_baseline",
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_transfer(
    image_size: int = config.IMAGE_SIZE,
    num_classes: int = config.NUM_CLASSES,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """EfficientNetB0 transfer model. Returns ``(model, base)``.

    The backbone is returned separately so the caller can unfreeze its top layers
    for Stage-2 fine-tuning. Stage 1 trains with the base frozen.
    """
    base = EfficientNetB0(include_top=False, weights="imagenet",
                          input_shape=(image_size, image_size, 3))
    base.trainable = False                       # Stage 1: freeze the backbone

    inputs = layers.Input((image_size, image_size, 3))
    x = make_augmentation()(inputs)
    x = preprocess_input(x)                      # EfficientNet expects raw [0,255]
    x = base(x, training=False)                  # nested 'efficientnetb0' (last conv: top_conv)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="efficientnet_transfer")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base
