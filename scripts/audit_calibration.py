"""Descriptive calibration audit for a saved CHURNIQ model.

Reconstructs the same group-disjoint test cohort used in churniq.model.train.
This NEVER fits a calibrator, changes thresholds, or tunes the model on held-out
data. It uses the SAVED FULL CANDIDATE artifact (trained across all time steps).
It does NOT constitute an out-of-time calibration assessment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from churniq.data import validate_prediction_data
from churniq.model import load


def calibration_table(y_true, probabilities, n_bins: int = 10):
    """Equal-width predicted-risk bins and expected calibration error."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("Inputs need equal, nonzero lengths.")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite numbers in [0, 1].")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Actual outcomes must be binary.")

    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.minimum(np.searchsorted(edges, p, side="right") - 1, n_bins - 1)
    rows = []
    expected_calibration_error = 0.0
    for i in range(n_bins):
        mask = idx == i
        count = int(mask.sum())
        predicted = float(p[mask].mean()) if count else None
        observed = float(y[mask].mean()) if count else None
        if count:
            expected_calibration_error += count / len(y) * abs(predicted - observed)
        rows.append({
            "risk_bin_start": round(float(edges[i]), 3),
            "risk_bin_end": round(float(edges[i + 1]), 3),
            "records": count,
            "mean_predicted_risk": predicted,
            "observed_no_purchase_rate": observed,
        })
    return rows, float(expected_calibration_error)


def audit_saved_model(data_path, model_path, bins=10):
    artifact = load(model_path)
    report = artifact["report"]
    if report.get("id_column") != "user_id" or "disjoint" not in report.get("split", ""):
        raise ValueError("Expected a REES46 model trained with --id user_id and a grouped holdout.")
    features = artifact["features"]
    needed = set(features) | {"user_id", "target_event"}
    df = pd.read_csv(data_path, usecols=lambda c: c in needed, low_memory=False)
    if needed - set(df):
        raise ValueError(f"Missing columns: {sorted(needed-set(df))}")
    if len(df) != report.get("source_rows"):
        raise ValueError("File row count differs from the saved training report.")

    groups = df["user_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    if len(train_idx) != report.get("train_rows") or len(test_idx) != report.get("test_rows"):
        raise ValueError("Holdout reconstruction disagrees with the saved training report.")
    if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
        raise AssertionError("Holdout contains customers used for model training.")

    test = df.iloc[test_idx]
    x = validate_prediction_data(test, features, artifact["numeric"], artifact["categorical"])
    truth = test["target_event"].astype(int).to_numpy()
    probabilities = artifact["pipeline"].predict_proba(x)[:, 1]
    table, ece = calibration_table(truth, probabilities, bins)
    prevalence = float(truth.mean())

    return {
        "audit": "descriptive_calibration_on_reconstructed_customer_group_holdout",
        "training_artifact": str(model_path),
        "customer_holdout_records": int(len(test)),
        "customer_holdout_unique_customers": int(groups.iloc[test_idx].nunique()),
        "observed_non_purchase_rate": prevalence,
        "average_predicted_risk": float(probabilities.mean()),
        "brier_score": float(brier_score_loss(truth, probabilities)),
        "constant_prevalence_brier_reference": float(np.mean((truth - prevalence) ** 2)),
        "log_loss": float(log_loss(truth, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "expected_calibration_error": ece,
        "equal_width_bins": table,
        "method_cautions": [
            "This is a descriptive reliability audit, not a fitted calibration model.",
            "Do not choose a decision threshold, model, or calibration method using this holdout.",
            "The model was trained across REES46 periods; this is NOT out-of-time calibration.",
            "SMOTE can distort risk probabilities, and temporal prevalence shift may further affect reliability.",
            "Reconstruction assumes the exact same original CSV and row order used for training.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect prediction reliability without refitting CHURNIQ.")
    parser.add_argument("--data", default="data/raw/rees46.csv")
    parser.add_argument("--model", default="artifacts/churniq_full_candidate.joblib")
    parser.add_argument("--output", default="data/reports/rees46_calibration_audit.json")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    print("Evaluating saved model probabilities on the original group-held-out customers...")
    audit = audit_saved_model(args.data, args.model, bins=args.bins)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Report saved: {output}")
    print(f"Holdout records: {audit['customer_holdout_records']:,}")
    print(f"Observed non-purchase rate: {audit['observed_non_purchase_rate']:.3f}")
    print(f"Mean predicted risk: {audit['average_predicted_risk']:.3f}")
    print(f"Brier score: {audit['brier_score']:.4f} "
          f"(constant-prevalence reference: {audit['constant_prevalence_brier_reference']:.4f})")
    print(f"ECE ({args.bins} bins): {audit['expected_calibration_error']:.4f}")
    print("DESCRIPTIVE ONLY. Do not use holdout information to select the model or threshold.")


if __name__ == "__main__":
    main()
