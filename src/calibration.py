"""
calibration.py
==============
Confidence-calibration and selective-prediction utilities — the "does the model
know when it is uncertain?" analysis.

* ``compute_ece``    — Expected Calibration Error + per-bin reliability stats.
* ``risk_coverage``  — selective-prediction trade-off (accuracy vs coverage when
  the model is allowed to abstain below a confidence threshold).

These are pure NumPy and dataset-free, so they are fully unit-testable.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class ECEResult(NamedTuple):
    """Expected Calibration Error and the per-bin stats behind it."""

    ece: float
    bin_centers: np.ndarray   # midpoint of each occupied confidence bin
    bin_accuracy: np.ndarray  # observed accuracy within the bin
    bin_confidence: np.ndarray  # mean predicted confidence within the bin
    bin_counts: np.ndarray    # number of samples in the bin


def compute_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> ECEResult:
    """Expected Calibration Error over ``n_bins`` equal-width confidence bins.

    Parameters
    ----------
    confidence:
        Predicted top-class probability for each sample, in ``[0, 1]``.
    correct:
        0/1 (or bool) array — was the prediction correct?
    n_bins:
        Number of equal-width bins spanning ``[0, 1]``.

    Returns
    -------
    ECEResult
        ``ece`` is the sample-weighted mean ``|accuracy - confidence|`` across
        occupied bins. Lower is better; 0 means predicted confidence exactly
        matches observed accuracy.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidence.shape != correct.shape:
        raise ValueError("confidence and correct must have the same shape")
    if confidence.size == 0:
        raise ValueError("inputs must be non-empty")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    centers, accs, confs, counts = [], [], [], []
    n = len(confidence)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum():
            acc_b = correct[mask].mean()
            conf_b = confidence[mask].mean()
            ece += (mask.sum() / n) * abs(acc_b - conf_b)
            centers.append((lo + hi) / 2)
            accs.append(acc_b)
            confs.append(conf_b)
            counts.append(int(mask.sum()))

    return ECEResult(
        ece=float(ece),
        bin_centers=np.array(centers),
        bin_accuracy=np.array(accs),
        bin_confidence=np.array(confs),
        bin_counts=np.array(counts),
    )


class RiskCoverage(NamedTuple):
    """Selective-prediction curve: accuracy on auto-decided cases vs coverage."""

    thresholds: np.ndarray
    coverage: np.ndarray   # fraction of cases with confidence >= threshold
    accuracy: np.ndarray   # accuracy among those auto-decided cases


def risk_coverage(
    confidence: np.ndarray,
    correct: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> RiskCoverage:
    """Compute the risk-coverage trade-off for selective prediction.

    For each confidence threshold ``t``, the model auto-decides cases with
    ``confidence >= t`` (routing the rest to a human) and we record how accurate
    it is on the cases it kept. As coverage drops, accuracy should rise — that
    monotone trend is what makes a confidence-tiered escalation policy valid.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.50, 0.995, 40)

    cov, acc = [], []
    for t in thresholds:
        keep = confidence >= t
        if keep.sum():
            cov.append(keep.mean())
            acc.append(correct[keep].mean())
        else:
            cov.append(0.0)
            acc.append(np.nan)
    return RiskCoverage(np.asarray(thresholds), np.asarray(cov), np.asarray(acc))
