"""Checks for aggregate reliability bin computation."""
import numpy as np
import pytest

from scripts.audit_calibration import calibration_table


def test_equal_width_bins_and_ece():
    labels = np.array([0, 1, 0, 1])
    probs = np.array([0.1, 0.2, 0.8, 1.0])
    rows, ece = calibration_table(labels, probs, n_bins=2)
    assert rows[0]["records"] == 2
    assert rows[1]["records"] == 2
    assert rows[0]["mean_predicted_risk"] == pytest.approx(0.15)
    assert rows[0]["observed_no_purchase_rate"] == pytest.approx(0.5)
    assert rows[1]["mean_predicted_risk"] == pytest.approx(0.9)
    assert rows[1]["observed_no_purchase_rate"] == pytest.approx(0.5)
    assert ece == pytest.approx(0.375)


def test_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="finite"):
        calibration_table([0, 1], [float("nan"), 0.5])


def test_rejects_nonbinary_target():
    with pytest.raises(ValueError, match="binary"):
        calibration_table([0, 2], [0.1, 0.9])
