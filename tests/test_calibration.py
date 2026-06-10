"""Tests for the calibration / selective-prediction utilities (pure NumPy, fast)."""

import numpy as np
import pytest

from src.calibration import compute_ece, risk_coverage


def test_perfectly_calibrated_has_low_ece():
    # In each bin, confidence == accuracy -> ECE should be (near) zero.
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.0, 1.0, size=10_000)
    # Draw correctness as a Bernoulli with p = confidence -> perfectly calibrated.
    correct = (rng.uniform(size=conf.size) < conf).astype(float)
    result = compute_ece(conf, correct, n_bins=10)
    assert result.ece < 0.03


def test_maximally_overconfident_has_high_ece():
    # Always 100% confident but always wrong -> ECE should be ~1.0.
    conf = np.ones(500)
    correct = np.zeros(500)
    result = compute_ece(conf, correct)
    assert result.ece == pytest.approx(1.0, abs=1e-9)


def test_ece_bins_are_consistent():
    conf = np.array([0.55, 0.65, 0.95, 0.99])
    correct = np.array([1, 0, 1, 1])
    result = compute_ece(conf, correct, n_bins=10)
    # ECE is a weighted average of |acc - conf|, so it stays in [0, 1].
    assert 0.0 <= result.ece <= 1.0
    assert result.bin_counts.sum() == conf.size


def test_ece_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        compute_ece(np.array([0.9, 0.8]), np.array([1]))


def test_ece_rejects_empty():
    with pytest.raises(ValueError):
        compute_ece(np.array([]), np.array([]))


def test_risk_coverage_accuracy_rises_as_coverage_falls():
    # Confidence correlates with correctness: higher thresholds keep better cases.
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.5, 1.0, size=5000)
    correct = (rng.uniform(size=conf.size) < conf).astype(float)
    rc = risk_coverage(conf, correct, thresholds=np.array([0.5, 0.7, 0.9]))
    # Coverage is monotonically non-increasing in threshold.
    assert rc.coverage[0] >= rc.coverage[1] >= rc.coverage[2]
    # Accuracy on retained cases is non-decreasing as we get stricter.
    assert rc.accuracy[0] <= rc.accuracy[2] + 1e-6
