"""
predict.py
==========
Single-image inference CLI for the released EfficientNetB0 model.

Examples
--------
    python app/predict.py path/to/scan.jpg
    python app/predict.py scan.jpg --model models/efficientnet_model.keras
    python app/predict.py scan.jpg --gradcam out_overlay.png

It prints the predicted tumour class, the model's confidence, and the full
per-class probability distribution. With ``--gradcam`` it also saves a Grad-CAM
overlay showing which region drove the prediction.

⚕️  Research / educational use only — not a medical device.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image

# Allow running as `python app/predict.py ...` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.utils import load_model, set_seed  # noqa: E402


def load_image(path: str, image_size: int = config.IMAGE_SIZE) -> np.ndarray:
    """Load one image as a raw-[0,255] float batch of shape ``(1, H, W, 3)``.

    The model normalizes internally, so we deliberately do **not** scale here.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    return np.expand_dims(np.asarray(img, dtype=np.float32), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a single brain MRI scan.")
    parser.add_argument("image", help="path to an MRI image (jpg/png/...)")
    parser.add_argument("--model", default=config.EFFNET_PATH, help="path to a .keras model")
    parser.add_argument("--gradcam", metavar="OUT.png", default=None,
                        help="also save a Grad-CAM overlay to this path")
    args = parser.parse_args()

    set_seed()
    model = load_model(args.model)
    batch = load_image(args.image)

    probs = model.predict(batch, verbose=0)[0]
    idx = int(probs.argmax())
    label = config.DISPLAY_NAMES.get(config.CLASS_NAMES[idx], config.CLASS_NAMES[idx])

    print(f"\nImage      : {args.image}")
    print(f"Prediction : {label}")
    print(f"Confidence : {probs[idx]:.1%}\n")
    print("Per-class probabilities:")
    for name, p in sorted(zip(config.CLASS_NAMES, probs, strict=True), key=lambda kv: -kv[1]):
        bar = "█" * int(round(p * 30))
        print(f"  {config.DISPLAY_NAMES[name]:<11} {p:6.1%}  {bar}")

    if args.gradcam:
        from src.gradcam import locate_last_conv, make_gradcam_heatmap, overlay_heatmap

        gc_layers, conv_idx, conv_name = locate_last_conv(model, batch)
        heatmap, _, _ = make_gradcam_heatmap(model, batch, gc_layers, conv_idx, pred_index=idx)
        overlay = overlay_heatmap(batch[0], heatmap)
        Image.fromarray(overlay).save(args.gradcam)
        print(f"\nGrad-CAM overlay (target layer '{conv_name}') saved to: {args.gradcam}")

    print("\n⚕️  Research/educational use only — not a medical device.")


if __name__ == "__main__":
    main()
