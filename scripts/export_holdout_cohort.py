"""Export an unlabelled REES46 cohort from customers held out of model training.

For demonstration only. Reconstructs the exact group holdout used by
churniq.model.train with seed=42, GroupShuffleSplit, test_size=0.20.
Only works if the artifact was trained on the same, unchanged CSV in the
same row order. Does not fabricate new customers or prospective outcomes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from churniq.model import load


def export_holdout_cohort(
    data_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    count: int = 100,
    period: int = 0,
) -> tuple[int, int]:
    if count < 1:
        raise ValueError("count must be positive.")
    artifact = load(model_path)
    report = artifact["report"]
    if report.get("id_column") != "user_id" or "disjoint" not in report.get("split", ""):
        raise ValueError("Model must have been trained with --id user_id using the grouped split.")
    features = artifact["features"]
    requested = set(features) | {"user_id", "time_step"}
    df = pd.read_csv(data_path, usecols=lambda column: column in requested, low_memory=False)

    missing = requested - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(df) != report.get("source_rows"):
        raise ValueError("CSV row count differs from the dataset used to train the model.")

    groups = df["user_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_indices, test_indices = next(splitter.split(df, groups=groups))
    train_ids = set(groups.iloc[train_indices])
    test_ids = set(groups.iloc[test_indices])
    if train_ids & test_ids:
        raise AssertionError("Holdout customers unexpectedly overlap with training customers.")

    subset = df.iloc[test_indices]
    subset = subset.loc[subset["time_step"] == period]
    if subset.empty:
        raise ValueError(f"No held-out customer rows in time_step={period}.")

    sample = subset.sample(n=min(count, len(subset)), random_state=42)
    output = sample.loc[:, ["user_id", *features]].copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return len(output), len(test_ids)


def main():
    parser = argparse.ArgumentParser(description="Make a label-free holdout upload cohort for CHURNIQ.")
    parser.add_argument("--data", default="data/raw/rees46.csv")
    parser.add_argument("--model", default="artifacts/churniq_rees46.joblib")
    parser.add_argument("--output", default="data/sample/rees46_heldout_cohort.csv")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--period", type=int, default=0, help="REES46 time_step; 0 is newest.")
    args = parser.parse_args()
    rows, heldout_customers = export_holdout_cohort(
        args.data, args.model, args.output, args.count, args.period
    )
    print(f"Saved {rows} label-free records to {args.output}.")
    print(f"Eligible holdout customers (all periods): {heldout_customers:,}.")
    print("These customer IDs were excluded from model fitting by the grouped split.")
    print("These are historical held-out records, NOT prospective new customer data.")


if __name__ == "__main__":
    main()
