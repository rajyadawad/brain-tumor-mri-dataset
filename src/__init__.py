"""
Brain Tumor MRI Classification — reusable pipeline.

The notebook (``notebooks/brain_tumor_classification.ipynb``) is the narrative;
this package is the same pipeline factored into importable, testable modules:

* :mod:`src.config`        — paths and hyperparameters (env-overridable).
* :mod:`src.utils`         — seeding, logging, model loading.
* :mod:`src.preprocessing` — augmentation factory.
* :mod:`src.data_loader`   — ``tf.data`` pipelines + class counts.
* :mod:`src.models`        — CNN baseline and EfficientNetB0 transfer model.
* :mod:`src.train`         — two-stage training CLI (with the H1/H2 fix).
* :mod:`src.evaluate`      — evaluate released checkpoints, emit metrics JSON.
* :mod:`src.gradcam`       — Keras-3-robust Grad-CAM.
* :mod:`src.calibration`   — ECE and selective-prediction utilities.
"""

__version__ = "1.0.0"
