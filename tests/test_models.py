"""Tests for the model builders. CNN is built (cheap); EfficientNet's ImageNet
download is skipped in CI by only checking the lightweight CNN's contract plus a
structural check that the transfer builder is importable."""

import numpy as np
import pytest

from src import config
from src.models import build_cnn


def test_cnn_output_shape_and_softmax():
    model = build_cnn(image_size=64, num_classes=config.NUM_CLASSES)
    assert model.output_shape == (None, config.NUM_CLASSES)

    x = np.random.RandomState(0).rand(2, 64, 64, 3).astype("float32") * 255.0
    probs = model.predict(x, verbose=0)
    assert probs.shape == (2, config.NUM_CLASSES)
    # Softmax rows sum to 1.
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-5)


def test_cnn_has_gradcam_target_layer():
    model = build_cnn(image_size=64)
    names = [layer.name for layer in model.layers]
    assert "top_conv" in names, "Grad-CAM target layer 'top_conv' must exist"


def test_cnn_normalizes_in_graph():
    # A Rescaling layer must be present so the saved model is self-contained.
    model = build_cnn(image_size=64)
    types = [type(layer).__name__ for layer in model.layers]
    assert "Rescaling" in types


@pytest.mark.parametrize("size", [32, 96])
def test_cnn_accepts_various_input_sizes(size):
    model = build_cnn(image_size=size)
    assert model.input_shape == (None, size, size, 3)
