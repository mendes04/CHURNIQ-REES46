"""Leakage-aware *exploratory* sigmoid calibration experiment for REES46.

A fresh model is fit on group-disjoint training customers (roughly 60%),
Platt sigmoid calibration is fit on separate customers (roughly 20%),
and raw/calibrated predictions are compared on the remaining customers (20%).

The data and the modelling options have been examined before; consequently
this is an exploratory split comparison and NOT a newly blinded thesis test.
No existing CHURNIQ model artifact is modified or overwritten.

Run:
  python -m scripts.experiment_calibration --data data/raw/rees46.csv --fast
  python -m scripts.experiment_calibration --data data/raw/rees46.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, StratifiedGroupKFold
from xgboost import XGBClassifier

from churniq.data import load_csv, prepare_training_data, validate_prediction_data, validate_target
from churniq.model import SEED, _pipeline
from scripts.audit_calibration import calibration_table


def group_partitions(df: pd.DataFrame):
    """Return disjoint 60/20/20 customer splits in training/calibration/test order."""
    if "user_id" not in df.columns:
        raise ValueError("REES46 user_id column is required.")
    groups = df.user_id.astype(str)
    if groups.nunique() < 30:
        raise ValueError("At least 30 distinct customers are required for a three-way split.")

    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    fitcal_idx, test_idx = next(outer.split(df, groups=groups))
    fitcal = df.iloc[fitcal_idx].copy()
    test = df.iloc[test_idx].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    train_idx, cal_idx = next(inner.split(fitcal, groups=fitcal.user_id.astype(str)))
    train = fitcal.iloc[train_idx].copy()
    calibration = fitcal.iloc[cal_idx].copy()

    train_ids = set(train.user_id.astype(str))
    calibration_ids = set(calibration.user_id.astype(str))
    test_ids = set(test.user_id.astype(str))
    if train_ids & calibration_ids or train_ids & test_ids or calibration_ids & test_ids:
        raise AssertionError("Customer overlap detected across experimental subsets.")

    for name, subset in (("training", train), ("calibration", calibration), ("test", test)):
        if subset.target_event.nunique() != 2:
            raise ValueError(f"{name} subset is missing a target class.")
    return train, calibration, test


def _logit(probability):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_sigmoid_calibrator(probability, y_true):
    """Fit Platt sigmoid on a dedicated calibration subset, never on testing data."""
    calibrator = LogisticRegression(max_iter=1000, random_state=SEED)
    calibrator.fit(_logit(probability), np.asarray(y_true, dtype=int))
    return calibrator


def describe_scores(y_true, probability, bins=10):
    p = np.asarray(probability, dtype=float)
    y = np.asarray(y_true, dtype=int)
    pred = (p >= 0.5).astype(int)
    _, ece = calibration_table(y, p, n_bins=bins)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece_10_bins": float(ece),
        "mean_predicted_risk": float(p.mean()),
        "observed_no_purchase_rate": float(y.mean()),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def run_experiment(df: pd.DataFrame, fast=False, cv=3, features=40):
    required = {"user_id", "target_event"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required fields: {sorted(required - set(df.columns))}")
    validate_target(df, "target_event")
    training, calibration, testing = group_partitions(df)

    xtrain, ytrain, excluded = prepare_training_data(training, "target_event", "user_id")
    model = XGBClassifier(
        random_state=SEED, eval_metric="logloss", tree_method="hist", n_jobs=2
    )
    pipe, numeric, categorical = _pipeline(xtrain, model, max_features=features)
    grid = (
        {"model__max_depth": [3], "model__learning_rate": [0.1],
         "model__n_estimators": [75], "model__subsample": [0.85]}
        if fast else
        {"model__max_depth": [3, 5], "model__learning_rate": [0.05, 0.1],
         "model__n_estimators": [120, 240], "model__subsample": [0.85]}
    )
    search = GridSearchCV(
        pipe, param_grid=grid, cv=StratifiedGroupKFold(
            n_splits=cv, shuffle=True, random_state=SEED
        ), scoring="f1", refit=True, error_score="raise", n_jobs=1
    )
    search.fit(xtrain, ytrain, groups=training.user_id.astype(str))
    fitted = search.best_estimator_

    def probs(subset):
        x = validate_prediction_data(subset, xtrain.columns.tolist(), numeric, categorical)
        return fitted.predict_proba(x)[:, 1]

    pcal = probs(calibration)
    calibrator = fit_sigmoid_calibrator(pcal, calibration.target_event)
    ptest = probs(testing)
    ptest_calibrated = calibrator.predict_proba(_logit(ptest))[:, 1]

    return {
        "experiment": "exploratory_group_disjoint_sigmoid_calibration",
        "target": "1 = no purchase in following four weeks",
        "fast": bool(fast),
        "train_rows": len(training),
        "calibration_rows": len(calibration),
        "test_rows": len(testing),
        "train_customers": int(training.user_id.nunique()),
        "calibration_customers": int(calibration.user_id.nunique()),
        "test_customers": int(testing.user_id.nunique()),
        "group_overlap": False,
        "selected_training_features": len(xtrain.columns),
        "selected_transformed_feature_count": int(
            fitted.named_steps["select"].get_support().sum()
        ),
        "dropped_columns": excluded,
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "training_cv_f1": float(search.best_score_),
        "calibration_method": "Platt sigmoid fitted on disjoint calibration customers",
        "calibration_sigmoid_coef": float(calibrator.coef_[0, 0]),
        "raw_test": describe_scores(testing.target_event, ptest),
        "calibrated_test": describe_scores(testing.target_event, ptest_calibrated),
        "warnings": [
            "This is an exploratory analysis: previous CHURNIQ evaluations have already examined the same underlying REES46 dataset.",
            "All subsets exclude each other's customer IDs, but their time periods are pooled: this is NOT a temporal generalisation test.",
            "Parameter search and feature selection use training folds only; calibration sees only the calibration group; test scores are descriptive.",
            "Do not select models, calibration methods, or thresholds using the test scores.",
            "Do not promote this experimental model to the dashboard; no production model was saved.",
            "Calibrated probabilities for the newest REES46 period may differ because its base rate shifts substantially.",
        ],
    }


def main():
    p = argparse.ArgumentParser(description="Three-way customer-disjoint sigmoid calibration experiment.")
    p.add_argument("--data", default="data/raw/rees46.csv")
    p.add_argument("--output", default="data/reports/rees46_calibration_experiment.json")
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--features", type=int, default=40)
    p.add_argument("--fast", action="store_true", help="Quick smoke test with one parameter setting.")
    args = p.parse_args()

    print("Running group-disjoint training / sigmoid calibration / evaluation...")
    report = run_experiment(load_csv(args.data), fast=args.fast, cv=args.cv, features=args.features)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report to: {destination}")
    print(f"Groups: {report['train_customers']:,} training | "
          f"{report['calibration_customers']:,} calibration | "
          f"{report['test_customers']:,} test")
    for key in ("raw_test", "calibrated_test"):
        metrics = report[key]
        print(f"{key}: Brier={metrics['brier']:.4f} "
              f"ECE={metrics['ece_10_bins']:.4f} "
              f"F1={metrics['f1']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
    print("Exploratory only; your existing dashboard model has not been changed.")


if __name__ == "__main__":
    main()
