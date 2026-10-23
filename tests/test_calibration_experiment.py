"""Tests for three-way REES46 calibration experiment's data separation."""
import numpy as np
import pandas as pd
import pytest

from scripts.experiment_calibration import (
    _logit, describe_scores, fit_sigmoid_calibrator, group_partitions,
)


def test_group_partitions_are_disjoint():
    records = [
        {"user_id": user, "time_step": step, "target_event": (user + step) % 2}
        for user in range(60)
        for step in (0, 1, 2)
    ]
    d = pd.DataFrame(records)
    train, calibration, test = group_partitions(d)
    a, b, c = (set(part.user_id) for part in (train, calibration, test))
    assert a.isdisjoint(b)
    assert a.isdisjoint(c)
    assert b.isdisjoint(c)
    assert len(a) + len(b) + len(c) == 60
    assert len(train) + len(calibration) + len(test) == len(d)


def test_sigmoid_outputs_valid_probabilities():
    p = np.array([0.01, 0.1, 0.2, 0.8, 0.9, 0.99] * 3)
    truth = np.array([0, 0, 0, 1, 1, 1] * 3)
    c = fit_sigmoid_calibrator(p, truth)
    calibrated = c.predict_proba(_logit(p))[:, 1]
    assert np.isfinite(calibrated).all()
    assert ((0 <= calibrated) & (calibrated <= 1)).all()
    report = describe_scores(truth, calibrated)
    assert 0 <= report["brier"] <= 1
    assert len(report["confusion_matrix"]) == 2


def test_partition_needs_customer_id():
    with pytest.raises(ValueError, match="user_id"):
        group_partitions(pd.DataFrame({"target_event": [0, 1]}))
