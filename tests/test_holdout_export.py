"""Tests for creating honest, label-free REES46 holdout upload examples."""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from scripts.export_holdout_cohort import export_holdout_cohort


def test_export_matches_grouped_training_holdout(tmp_path):
    records = []
    for customer in range(40):
        for step in range(3):
            records.append({
                "user_id": customer,
                "time_step": step,
                "feature_x": customer * 0.1 + step,
                "target_event": customer % 2,
                "row_id": len(records) + 1,
            })
    df = pd.DataFrame(records)
    csv = tmp_path / "train.csv"
    model = tmp_path / "model.joblib"
    output = tmp_path / "upload.csv"
    df.to_csv(csv, index=False)
    joblib.dump({
        "features": ["feature_x"],
        "report": {
            "id_column": "user_id",
            "source_rows": len(df),
            "split": "80/20 of unique customer IDs, disjoint across train/test; grouped CV on training records",
        },
    }, model)

    n, groups = export_holdout_cohort(csv, model, output, count=100, period=0)
    result = pd.read_csv(output)
    assert n == groups
    assert set(result.columns) == {"user_id", "feature_x"}
    assert result["user_id"].is_unique
    assert len(result) == 8

    labels = df.user_id.astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=labels))
    train_ids = set(df.iloc[train_idx].user_id)
    test_ids = set(df.iloc[test_idx].user_id)
    assert set(result.user_id).issubset(test_ids)
    assert not set(result.user_id).intersection(train_ids)
