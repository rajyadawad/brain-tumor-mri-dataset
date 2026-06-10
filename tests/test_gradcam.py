"""Tests for the Keras-3-robust Grad-CAM core, using a tiny dummy CNN (no dataset,
no pretrained download)."""

import numpy as np
import pytest
from tensorflow.keras import layers, models

from src.gradcam import locate_last_conv, make_gradcam_heatmap, overlay_heatmap


def _tiny_cnn(size=32, classes=4):
    return models.Sequential([
        layers.Input((size, size, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(4, 3, padding="same", activation="relu"),
        layers.Conv2D(8, 3, padding="same", activation="relu", name="top_conv"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(classes, activation="softmax"),
    ], name="tiny")


def test_locate_last_conv_picks_last_4d_layer():
    model = _tiny_cnn()
    sample = np.zeros((1, 32, 32, 3), dtype="float32")
    gc_layers, conv_idx, conv_name = locate_last_conv(model, sample)
    assert conv_name == "top_conv"
    assert 0 <= conv_idx < len(gc_layers)


def test_gradcam_returns_normalized_heatmap():
    model = _tiny_cnn(size=32)
    img = np.random.RandomState(0).rand(1, 32, 32, 3).astype("float32") * 255.0
    gc_layers, conv_idx, _ = locate_last_conv(model, img)
    heatmap, idx, probs = make_gradcam_heatmap(model, img, gc_layers, conv_idx)

    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-6
    assert 0 <= idx < 4
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-5)


def test_overlay_returns_uint8_rgb_matching_image():
    img = (np.random.RandomState(1).rand(48, 48, 3) * 255).astype("uint8")
    heatmap = np.random.RandomState(2).rand(8, 8).astype("float32")
    overlay = overlay_heatmap(img, heatmap)
    assert overlay.shape == (48, 48, 3)
    assert overlay.dtype == np.uint8


def test_gradcam_rejects_non_conv_model():
    mlp = models.Sequential([
        layers.Input((10,)),
        layers.Dense(4, activation="softmax"),
    ])
    with pytest.raises(ValueError):
        locate_last_conv(mlp, np.zeros((1, 10), dtype="float32"))
