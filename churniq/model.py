"""Reproducible training and prediction for the thesis's pre-labelled REES46 table.

Important: class label 1 = no interaction during the subsequent month.
SMOTE and feature selection are INSIDE cross-validation pipelines.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, learning_curve
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from churniq.data import prepare_training_data, validate_prediction_data

SEED = 42


def _pipeline(x: pd.DataFrame, estimator, max_features: int):
    numeric = x.select_dtypes(include=np.number).columns.tolist()
    cats = [col for col in x.columns if col not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", SKPipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric))
    if cats:
        transformers.append(("cat", SKPipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=12, sparse_output=False))
        ]), cats))
    prep = ColumnTransformer(transformers, sparse_threshold=0, verbose_feature_names_out=True)
    k = min(max_features, len(x.columns))
    pipe = Pipeline([
        ("prep", prep),
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("smote", SMOTE(random_state=SEED, k_neighbors=3)),
        ("model", estimator)
    ])
    return pipe, numeric, cats


def _scores(y, prob, threshold=0.5):
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def train(df: pd.DataFrame, target: str, identifier: str | None = None,
          fast: bool = False, cv: int = 3, max_features: int = 40):
    x, y, dropped = prepare_training_data(df, target, identifier)
    if y.value_counts().min() < cv * 4:
        raise ValueError("Not enough minority-class samples for safe CV/SMOTE; reduce folds or use more records.")
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.20, random_state=SEED, stratify=y)
    estimators = {
        "Logistic Regression": LogisticRegression(max_iter=1200, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=70 if fast else 180, random_state=SEED, n_jobs=-1),
        "Default XGBoost": XGBClassifier(n_estimators=70 if fast else 180, random_state=SEED,
                                         eval_metric="logloss", tree_method="hist", n_jobs=2),
        "Tuned XGBoost": XGBClassifier(random_state=SEED, eval_metric="logloss", tree_method="hist", n_jobs=2),
    }
    scores = {}
    fitted = {}
    best_parameters = {}
    splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    for name, estimator in estimators.items():
        pipe, numeric, cats = _pipeline(xtrain, estimator, max_features)
        if name == "Tuned XGBoost":
            grid = ({"model__max_depth": [3], "model__learning_rate": [0.1], "model__n_estimators": [75]}
                    if fast else {"model__max_depth": [3, 5],
                                  "model__learning_rate": [0.05, 0.1],
                                  "model__n_estimators": [120, 240],
                                  "model__subsample": [0.85]})
            search = GridSearchCV(pipe, grid, scoring="f1", cv=splitter, n_jobs=1,
                                  refit=True, error_score="raise")
            search.fit(xtrain, ytrain)
            pipe = search.best_estimator_
            best_parameters = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
            best_parameters["cross_validation_f1"] = float(search.best_score_)
        else:
            pipe.fit(xtrain, ytrain)
        prob = pipe.predict_proba(xtest)[:, 1]
        scores[name] = _scores(ytest, prob)
        fitted[name] = pipe
    winner = "Tuned XGBoost"
    model = fitted[winner]
    feature_names = model.named_steps["prep"].get_feature_names_out()
    selected_names = feature_names[model.named_steps["select"].get_support()].tolist()
    learning = None
    if not fast:
        train_sizes, train_scores, val_scores = learning_curve(
            model, xtrain, ytrain, train_sizes=[0.3, 0.5, 0.75, 1.0],
            scoring="f1", cv=splitter, n_jobs=1, shuffle=True, random_state=SEED,
            error_score=np.nan)
        learning = {"train_sizes": train_sizes.tolist(),
                    "train_f1": np.nanmean(train_scores,axis=1).tolist(),
                    "validation_f1": np.nanmean(val_scores,axis=1).tolist()}
    report = {
        "learning_curve": learning,
        "training_class_counts_before_smote": {str(k):int(v) for k,v in ytrain.value_counts().items()},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_target_definition": "1 = no interaction in subsequent month; 0 = returned",
        "source_rows": int(len(df)),
        "train_rows": int(len(xtrain)),
        "test_rows": int(len(xtest)),
        "churn_rate": float(y.mean()),
        "features": x.columns.tolist(),
        "selected_transformed_features": selected_names,
        "numeric": numeric,
        "categorical": cats,
        "excluded_columns": dropped,
        "target_column": target,
        "id_column": identifier,
        "model_used": winner,
        "best_by_test_f1": max(scores, key=lambda k: scores[k]["f1"]),
        "best_params": best_parameters,
        "scores": scores,
        "split": "80/20 stratified random holdout; CV limited to training set",
        "cv_folds": cv,
        "warning": "The held-out test set must NOT be used repeatedly for model/threshold selection. Confirm time boundaries and customer overlap.",
    }
    roc = {}
    for key, pipe in fitted.items():
        probabilities = pipe.predict_proba(xtest)[:, 1]
        fpr, tpr, _ = roc_curve(ytest, probabilities)
        idx = np.linspace(0, len(fpr) - 1, min(300, len(fpr)), dtype=int)
        roc[key] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}
    report["roc_curves"] = roc
    artifact = {"pipeline": model, "report": report, "features": x.columns.tolist(),
                "numeric": numeric, "categorical": cats, "selected_names": selected_names}
    return artifact


def save(artifact, destination: str | Path):
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    return path


def load(path: str | Path):
    return joblib.load(path)


def predict(artifact, df: pd.DataFrame, threshold: float = 0.5, id_column: str | None = None):
    x = validate_prediction_data(df, artifact["features"], artifact["numeric"], artifact["categorical"])
    probability = artifact["pipeline"].predict_proba(x)[:, 1]
    if id_column and id_column in df.columns:
        ids = df[id_column].astype(str).to_numpy()
    else:
        ids = np.array([f"ROW-{i + 1:06d}" for i in range(len(df))])
    result = pd.DataFrame({
        "customer_id": ids,
        "predicted_inactivity_risk": probability,
        "predicted_inactive_next_month": (probability >= threshold).astype(int),
        "risk_band": pd.cut(probability, bins=[-0.001, 0.4, 0.7, 1.001],
                            labels=["Low", "Moderate", "High"], right=False).astype(str),
    })
    result["predicted_inactivity_risk_pct"] = (result["predicted_inactivity_risk"] * 100).round(1)
    return result


def transform_selected(artifact, df: pd.DataFrame):
    x = validate_prediction_data(df, artifact["features"], artifact["numeric"], artifact["categorical"])
    pipe = artifact["pipeline"]
    features = pipe.named_steps["prep"].transform(x)
    return pipe.named_steps["select"].transform(features)
