"""SHAP explanations for a fitted tuned XGBoost, in *log-odds* units."""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap
from churniq.model import transform_selected


def explain(artifact, raw: pd.DataFrame, max_rows: int = 300):
    if raw.empty:
        raise ValueError("Nothing to explain")
    selected = transform_selected(artifact, raw.head(max_rows))
    model = artifact["pipeline"].named_steps["model"]
    explainer = shap.TreeExplainer(model, model_output="raw")
    raw_exp = explainer(selected)
    vals = np.asarray(raw_exp.values)
    if vals.ndim == 3:
        vals = vals[:, :, -1]
    expected = np.asarray(raw_exp.base_values)
    if expected.ndim > 1:
        expected = expected[:, -1]
    if expected.ndim == 0:
        expected = np.repeat(float(expected), len(vals))
    return {"values": vals, "base_values": expected,
            "feature_values": selected, "feature_names": artifact["selected_names"],
            "units": "log-odds (positive raises inactivity risk)"}


def top_local(shap_result, index: int, limit: int = 12):
    s = shap_result["values"][index]
    inds = np.argsort(-np.abs(s))[:limit]
    return pd.DataFrame({"feature": np.asarray(shap_result["feature_names"])[inds],
                         "contribution_log_odds": s[inds],
                         "model_input_value": np.asarray(shap_result["feature_values"])[index, inds]})


def top_global(shap_result, limit: int = 16):
    means = np.abs(shap_result["values"]).mean(axis=0)
    inds = np.argsort(-means)[:limit]
    return pd.DataFrame({"feature": np.asarray(shap_result["feature_names"])[inds],
                         "mean_abs_shap_log_odds": means[inds]})
