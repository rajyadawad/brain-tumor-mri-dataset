"""
gradcam.py
==========
Grad-CAM explainability, robust to Keras 3 and to models that wrap a nested
pretrained backbone (EfficientNet).

Why not the textbook recipe? Under Keras 3 the usual
``tf.keras.Model(model.input, [conv.output, model.output])`` fails with
"Output ... is not connected to inputs" because a nested backbone keeps its own
internal graph. Instead we run a short layer-by-layer forward pass and capture
the last conv feature map directly from the live computation — which is also
architecture-agnostic (it works for the plain CNN too).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def _callable_layers(model: tf.keras.Model) -> list[tf.keras.layers.Layer]:
    """Top-level layers excluding the (non-callable) InputLayer."""
    return [layer for layer in model.layers
            if not isinstance(layer, tf.keras.layers.InputLayer)]


def locate_last_conv(model: tf.keras.Model, sample: np.ndarray) -> tuple[list, int, str]:
    """Trace one forward pass; return ``(layers, conv_idx, conv_name)`` for the
    deepest layer that emits a 4-D ``(H, W, C)`` feature map.

    Works for plain CNNs and for models wrapping a nested pretrained backbone.
    Raises a clear error for non-convolutional (unsupported) models.
    """
    layers = _callable_layers(model)
    x = tf.convert_to_tensor(sample, tf.float32)
    conv_idx = None
    for i, layer in enumerate(layers):
        x = layer(x, training=False)
        if len(x.shape) == 4:
            conv_idx = i
    if conv_idx is None:
        raise ValueError(
            f"Grad-CAM needs a convolutional feature map, but model {model.name!r} has no "
            "4-D layer output. Grad-CAM supports CNN-based models only."
        )
    return layers, conv_idx, layers[conv_idx].name


def make_gradcam_heatmap(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    gc_layers: list,
    conv_idx: int,
    pred_index: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Grad-CAM heatmap for a single raw-[0,255] image batch of shape ``(1, H, W, 3)``.

    ``gc_layers`` and ``conv_idx`` come from :func:`locate_last_conv` (resolve once,
    reuse). Returns ``(heatmap[h, w] in [0, 1], predicted_index, probability_vector)``.
    """
    with tf.GradientTape() as tape:
        x = tf.convert_to_tensor(img_batch, tf.float32)
        for layer in gc_layers[: conv_idx + 1]:        # input -> last conv feature map
            x = layer(x, training=False)
        conv_out = x
        tape.watch(conv_out)
        for layer in gc_layers[conv_idx + 1:]:         # feature map -> class probabilities
            x = layer(x, training=False)
        probs = x
        if pred_index is None:
            pred_index = int(tf.argmax(probs[0]))
        class_channel = probs[:, pred_index]

    grads = tape.gradient(class_channel, conv_out)         # dScore / dFeatureMap
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))         # importance per channel
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)    # weighted feature map
    heatmap = tf.nn.relu(heatmap)                          # keep positive evidence
    maxv = tf.reduce_max(heatmap)
    if maxv > 0:
        heatmap = heatmap / maxv                           # normalize to [0, 1]
    return heatmap.numpy(), int(pred_index), probs.numpy()[0]


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = "jet",
) -> np.ndarray:
    """Resize ``heatmap`` to ``image`` and alpha-blend it as an RGB overlay.

    ``image`` is expected in ``[0, 255]``; returns a uint8 RGB array.
    """
    import matplotlib.cm as cm
    from PIL import Image as PILImage

    h, w = image.shape[:2]
    hm = PILImage.fromarray(np.uint8(255 * heatmap)).resize((w, h))
    hm = np.asarray(hm, dtype=np.float32) / 255.0
    colored = cm.get_cmap(colormap)(hm)[..., :3]           # (h, w, 3) in [0, 1]

    base = image.astype(np.float32)
    if base.max() <= 1.0:
        base = base * 255.0
    out = (1 - alpha) * base + alpha * (colored * 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)
