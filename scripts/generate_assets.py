"""
generate_assets.py
==================
Regenerate every figure embedded in the README, straight from the released
checkpoints in ``models/`` and the held-out ``Dataset/Testing`` set. Also rewrites
``results/metrics/saved_model_eval.json`` so the numbers and the pictures always
agree.

Run from the repo root::

    python scripts/generate_assets.py

Outputs land in ``results/figures/`` (canonical) and the hero images are copied
into ``assets/`` for the README banner/showcase.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config  # noqa: E402
from src.calibration import compute_ece, risk_coverage  # noqa: E402
from src.data_loader import build_datasets, count_images_per_class  # noqa: E402
from src.gradcam import locate_last_conv, make_gradcam_heatmap, overlay_heatmap  # noqa: E402
from src.utils import get_logger, load_model, set_seed  # noqa: E402

logger = get_logger("generate_assets")
DISPLAY = config.DISPLAY_NAMES
PALETTE = {"effnet": "#1565c0", "cnn": "#90a4ae"}


def collect(test_ds):
    """Materialise the unshuffled test set as (uint8 images, int labels)."""
    images, labels = [], []
    for xb, yb in test_ds:
        images.append(xb.numpy().astype("uint8"))
        labels.append(yb.numpy())
    return np.concatenate(images), np.concatenate(labels)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_confusion_matrix(y_true, y_pred, class_names, out):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    labels = [DISPLAY[c] for c in class_names]
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("EfficientNetB0 — Confusion Matrix (test set)", fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_model_comparison(metrics_cnn, metrics_eff, out):
    keys = ["test_accuracy", "macro_precision", "macro_recall", "macro_f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    b1 = ax.bar(x - w / 2, [metrics_cnn[k] for k in keys], w,
                label="CNN baseline", color=PALETTE["cnn"], edgecolor="black")
    b2 = ax.bar(x + w / 2, [metrics_eff[k] for k in keys], w,
                label="EfficientNetB0", color=PALETTE["effnet"], edgecolor="black")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.2f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (macro-averaged)")
    ax.set_title("Model Comparison — CNN vs EfficientNetB0", fontweight="bold")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_reliability(conf, correct, out):
    res = compute_ece(conf, correct.astype(float))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
    ax1.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax1.bar(res.bin_confidence, res.bin_accuracy, width=0.09, alpha=0.25, color="#1565c0")
    ax1.plot(res.bin_confidence, res.bin_accuracy, "o-", color="#1565c0", label="EfficientNetB0")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("mean predicted confidence"); ax1.set_ylabel("observed accuracy")
    ax1.set_title(f"Reliability Diagram (ECE = {res.ece:.3f})", fontweight="bold")
    ax1.legend(loc="upper left")

    gap = res.bin_confidence - res.bin_accuracy
    colors = ["indianred" if g > 0 else "seagreen" for g in gap]
    ax2.bar(range(len(gap)), gap, color=colors)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(range(len(res.bin_centers)),
                   [f"{c:.0%}" for c in res.bin_centers], rotation=45)
    ax2.set_title("Confidence − Accuracy per bin\n(red = over-confident, green = under-confident)")
    ax2.set_xlabel("confidence bin"); ax2.set_ylabel("gap")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
    return res.ece


def fig_risk_tiers(conf, correct, out):
    def tier(c):
        return "High" if c >= 0.90 else ("Medium" if c >= 0.70 else "Low")

    order = ["High", "Medium", "Low"]
    colors = {"High": "#2e7d32", "Medium": "#f9a825", "Low": "#c62828"}
    tiers = np.array([tier(c) for c in conf])
    counts = {t: int((tiers == t).sum()) for t in order}
    acc = {t: (correct[tiers == t].mean() if counts[t] else 0.0) for t in order}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.7))
    bars = ax1.bar(order, [counts[t] for t in order],
                   color=[colors[t] for t in order], edgecolor="black")
    for b, t in zip(bars, order, strict=True):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
                 f"{counts[t]}\n({100*counts[t]/len(conf):.0f}%)", ha="center", fontsize=10)
    ax1.set_title("Case volume by confidence tier"); ax1.set_ylabel("test scans")
    ax1.set_ylim(0, max(counts.values()) * 1.2)

    bars2 = ax2.bar(order, [100 * acc[t] for t in order],
                    color=[colors[t] for t in order], edgecolor="black")
    ax2.axhline(100 * correct.mean(), ls="--", color="navy",
                label=f"overall {100*correct.mean():.1f}%")
    for b, t in zip(bars2, order, strict=True):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                 f"{100*acc[t]:.1f}%", ha="center", fontsize=10)
    ax2.set_title("Observed accuracy by tier"); ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(0, 108); ax2.legend(loc="lower left")
    for a in (ax1, ax2):
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Risk Stratification by Model Confidence", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_selective_prediction(conf, correct, out):
    rc = risk_coverage(conf, correct.astype(float))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(rc.coverage, rc.accuracy, "o-", color="darkorange", ms=4)
    ax.set_xlabel("coverage (fraction of scans auto-decided)")
    ax.set_ylabel("accuracy on auto-decided scans")
    ax.set_title("Selective Prediction — accuracy vs coverage", fontweight="bold")
    ax.invert_xaxis(); ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_sample_predictions(images, y_true, y_pred, conf, class_names, out, n=8):
    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(images), size=n, replace=False)
    cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    for ax, i in zip(axes.ravel(), idx, strict=False):
        ax.imshow(images[i].astype("uint8")); ax.axis("off")
        ok = y_pred[i] == y_true[i]
        ax.set_title(
            f"pred: {DISPLAY[class_names[y_pred[i]]]} ({conf[i]:.0%})\n"
            f"true: {DISPLAY[class_names[y_true[i]]]}",
            color="#2e7d32" if ok else "#c62828", fontsize=10)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Sample Test Predictions (EfficientNetB0)", fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_gradcam_showcase(model, images, y_true, y_pred, conf, class_names, out):
    gc_layers, conv_idx, conv_name = locate_last_conv(model, images[:1].astype("float32"))
    logger.info("Grad-CAM target layer: %s", conv_name)
    n = len(class_names)
    fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 6.2))
    for col, cls_idx in enumerate(range(n)):
        # pick the most confident CORRECT example of this class
        mask = (y_true == cls_idx) & (y_pred == cls_idx)
        cand = np.where(mask)[0]
        i = cand[np.argmax(conf[cand])] if len(cand) else int(np.argmax(y_true == cls_idx))
        img = images[i].astype("float32")[None]
        heatmap, _, _ = make_gradcam_heatmap(model, img, gc_layers, conv_idx, pred_index=cls_idx)
        overlay = overlay_heatmap(images[i], heatmap)

        axes[0, col].imshow(images[i].astype("uint8")); axes[0, col].axis("off")
        axes[0, col].set_title(DISPLAY[class_names[cls_idx]], fontweight="bold")
        axes[1, col].imshow(overlay); axes[1, col].axis("off")
        axes[1, col].set_title(f"Grad-CAM ({conf[i]:.0%})", fontsize=10)
    axes[0, 0].set_ylabel("MRI", rotation=0, labelpad=30)
    fig.suptitle("Grad-CAM — where EfficientNetB0 looks (confident, correct cases)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_banner(acc, ece, out):
    fig, ax = plt.subplots(figsize=(12, 3.1))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#0d1b2a"))
    ax.text(0.5, 0.66, "Brain Tumor MRI Classification",
            ha="center", va="center", color="white", fontsize=28, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.36, "Explainable Deep Learning  ·  EfficientNetB0  ·  Grad-CAM  ·  "
            "Calibration  ·  Clinical Decision Support",
            ha="center", va="center", color="#90caf9", fontsize=13, transform=ax.transAxes)
    ax.text(0.5, 0.12, f"4-class · {acc:.1%} test accuracy · ECE {ece:.3f}",
            ha="center", va="center", color="#cfd8dc", fontsize=12, transform=ax.transAxes)
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor="#0d1b2a"); plt.close(fig)


def macro_metrics(y_true, y_pred, conf):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    correct = (y_pred == y_true).astype(float)
    return {
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "mean_confidence": round(float(conf.mean()), 4),
        "ece_10bin": round(compute_ece(conf, correct).ece, 4),
        "n_test_errors": int((y_pred != y_true).sum()),
    }


def main() -> None:
    set_seed()
    import tensorflow as tf

    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.METRICS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.PROJECT_ROOT, "assets"), exist_ok=True)

    logger.info("Building test dataset...")
    _, _, test_ds, class_names = build_datasets()
    images, y_true = collect(test_ds)
    logger.info("Test set: %d images", len(images))

    logger.info("Loading models...")
    effnet = load_model(config.EFFNET_PATH)
    cnn = load_model(config.CNN_PATH)

    logger.info("Predicting (EfficientNet)...")
    eff_probs = effnet.predict(images, batch_size=config.BATCH_SIZE, verbose=0)
    logger.info("Predicting (CNN)...")
    cnn_probs = cnn.predict(images, batch_size=config.BATCH_SIZE, verbose=0)

    eff_pred, eff_conf = eff_probs.argmax(1), eff_probs.max(1)
    cnn_pred, cnn_conf = cnn_probs.argmax(1), cnn_probs.max(1)
    eff_correct = (eff_pred == y_true).astype(float)

    m_eff = macro_metrics(y_true, eff_pred, eff_conf)
    m_cnn = macro_metrics(y_true, cnn_pred, cnn_conf)
    logger.info("EfficientNet acc=%.4f | CNN acc=%.4f",
                m_eff["test_accuracy"], m_cnn["test_accuracy"])

    F = config.FIGURES_DIR
    fig_confusion_matrix(y_true, eff_pred, class_names, os.path.join(F, "confusion_matrix.png"))
    fig_model_comparison(m_cnn, m_eff, os.path.join(F, "model_comparison.png"))
    ece = fig_reliability(eff_conf, eff_correct, os.path.join(F, "reliability_diagram.png"))
    fig_risk_tiers(eff_conf, eff_correct, os.path.join(F, "risk_tier_dashboard.png"))
    fig_selective_prediction(eff_conf, eff_correct, os.path.join(F, "selective_prediction.png"))
    fig_sample_predictions(images, y_true, eff_pred, eff_conf, class_names,
                           os.path.join(F, "sample_predictions.png"))
    fig_gradcam_showcase(effnet, images, y_true, eff_pred, eff_conf, class_names,
                         os.path.join(F, "gradcam_showcase.png"))
    fig_banner(m_eff["test_accuracy"], ece, os.path.join(F, "banner.png"))

    # Copy the banner into assets/ (README chrome). All data figures are embedded
    # directly from results/figures/ to avoid duplicating large PNGs.
    assets = os.path.join(config.PROJECT_ROOT, "assets")
    shutil.copy(os.path.join(F, "banner.png"), os.path.join(assets, "banner.png"))

    # Rewrite the metrics JSON so numbers and figures agree.
    payload = {
        "description": (
            "Independent evaluation of the RELEASED checkpoints in models/ on "
            "Dataset/Testing (1,600 images), generated by scripts/generate_assets.py. "
            "These are the numbers quoted in the README and rendered in results/figures/."
        ),
        "environment": {
            "tensorflow": tf.__version__,
            "image_size": config.IMAGE_SIZE,
            "batch_size": config.BATCH_SIZE,
            "seed": config.SEED,
        },
        "classes": list(class_names),
        "train_counts": count_images_per_class(config.TRAIN_DIR),
        "test_counts": count_images_per_class(config.TEST_DIR),
        "released_checkpoints": {
            "models/efficientnet_model.keras": m_eff,
            "models/cnn_model.keras": m_cnn,
        },
    }
    out = os.path.join(config.METRICS_DIR, "saved_model_eval.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s", out)
    logger.info("Done. Figures in %s", F)


if __name__ == "__main__":
    main()
