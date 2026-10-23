"""Unit tests for chronological REES46 holdout construction."""
import pandas as pd
import pytest

from scripts.evaluate_temporal import temporal_partition


def test_newest_period_is_excluded_from_historical_training():
    data = pd.DataFrame({
        "user_id": [10, 10, 10, 11, 11, 11],
        "time_step": [0, 1, 2, 0, 1, 2],
        "target_event": [1, 0, 1, 0, 1, 0],
        "feature": [0, 1, 2, 3, 4, 5],
    })
    older, newest = temporal_partition(data)
    assert len(older) == 4
    assert len(newest) == 2
    assert set(older.time_step) == {1, 2}
    assert set(newest.time_step) == {0}
    assert not set(older.index).intersection(newest.index)


def test_rejects_duplicate_customer_period():
    data = pd.DataFrame({
        "user_id": [1, 1, 2, 3],
        "time_step": [0, 0, 1, 0],
        "target_event": [0, 1, 0, 1],
    })
    with pytest.raises(ValueError, match="Duplicate"):
        temporal_partition(data)
