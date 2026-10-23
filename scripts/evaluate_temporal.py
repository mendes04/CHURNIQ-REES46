"""Out-of-time validation for REES46's four-week non-purchase target.

The source's time_step=0 is most recent; steps 1..6 are older.
Train and tune models ONLY on older periods, then score period 0 exactly once.

This is a separate robustness experiment, NOT evaluation of a model already
trained on the full CSV (which would leak the newest period). Customer overlap
across older and later periods is permissible here: this tests future behaviour
of known or emerging customers, not generalisation to unseen customer IDs.
The helper train() uses a grouped internal split to pick fitting records and
to report an independent historic-period validation score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from churniq.data import load_csv, validate_prediction_data
from churniq.model import train


def temporal_partition(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"user_id", "time_step", "target_event"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing REES46 fields: {missing}")
    periods = pd.to_numeric(df["time_step"], errors="raise")
    if periods.isna().any() or (periods < 0).any():
        raise ValueError("time_step contains invalid period indices.")
    if df.duplicated(["user_id", "time_step"]).any():
        raise ValueError("Duplicate user_id/time_step pairs found.")
    older = df.loc[periods > 0].copy()
    newest = df.loc[periods == 0].copy()
    if older.empty or newest.empty:
        raise ValueError("Expected older REES46 periods (>0) and latest period 0.")
    if older["target_event"].nunique() != 2 or newest["target_event"].nunique() != 2:
        raise ValueError("Both older and newest periods must contain both target classes.")
    return older, newest


def evaluate_temporal(
    df: pd.DataFrame,
    *,
    fast: bool = False,
    cv: int = 3,
    features: int = 40,
) -> dict:
    older, newest = temporal_partition(df)

    # Critically, this training function never sees newest-period rows.
    artifact = train(older, target="target_event", identifier="user_id",
                     fast=fast, cv=cv, max_features=features)
    x = validate_prediction_data(newest, artifact["features"],
                                 artifact["numeric"], artifact["categorical"])
    truth = newest["target_event"].astype(int)
    prob = artifact["pipeline"].predict_proba(x)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, prob)),
        "average_precision": float(average_precision_score(truth, prob)),
        "brier": float(brier_score_loss(truth, prob)),
        "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1]).tolist(),
    }

    return {
        "experiment": "out_of_time_rees46_period_0",
        "target_definition": "1 = no purchase in following four weeks; 0 = purchased",
        "older_periods": sorted(older["time_step"].unique().tolist()),
        "test_period": 0,
        "older_rows": len(older),
        "test_rows": len(newest),
        "older_unique_customers": int(older["user_id"].nunique()),
        "test_unique_customers": int(newest["user_id"].nunique()),
        "older_non_purchase_rate": float(older["target_event"].mean()),
        "test_non_purchase_rate": float(newest["target_event"].mean()),
        "training_procedure": artifact["report"]["split"],
        "internal_historical_holdout": artifact["report"]["scores"]["Tuned XGBoost"],
        "selected_tuning_parameters": artifact["report"]["best_params"],
        "temporal_test": metrics,
        "fast": fast,
        "cautions": [
            "The fitted model uses only the training portion of older-period customers after its internal group split.",
            "Never use this newest-period result to choose hyperparameters or thresholds.",
            "Temporal testing permits the same person in an older training period and a newer test period, but no future-period rows are supplied to model fitting.",
            "The risk scores are not necessarily calibrated; the SMOTE pipeline can affect probability reliability.",
            "Do not compare group-holdout and out-of-time metrics as if they used the same test cohort.",
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate REES46 generalisation from past periods to latest period.")
    p.add_argument("--data", default="data/raw/rees46.csv")
    p.add_argument("--output", default="data/reports/rees46_temporal_evaluation.json")
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--features", type=int, default=40)
    p.add_argument("--fast", action="store_true", help="Quick pipeline check, not final tuning.")
    args = p.parse_args()
    print("Loading labelled REES46 data...")
    df = load_csv(args.data)
    older, newest = temporal_partition(df)
    print(f"Historical rows (time_step 1+): {len(older):,}")
    print(f"Latest-period rows (time_step 0): {len(newest):,}")
    print("Training exclusively on historical periods, tuning within historical training folds...")
    result = evaluate_temporal(df, fast=args.fast, cv=args.cv, features=args.features)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Temporal evaluation report saved: {path}")
    print("Historic internal holdout F1:",
          f"{result['internal_historical_holdout']['f1']:.4f}")
    m = result["temporal_test"]
    print(f"Newer-period F1={m['f1']:.4f} Recall={m['recall']:.4f} "
          f"ROC-AUC={m['roc_auc']:.4f} Brier={m['brier']:.4f}")
    print("Note: this model has NOT been trained on time_step=0.")


if __name__ == "__main__":
    main()
