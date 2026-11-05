"""Export thesis-ready global and local XGBoost/SHAP explanation evidence.

Reads an EXISTING trained CHURNIQ artifact and a local, unlabelled held-out
customer cohort. Does not fit, calibrate, overwrite, or promote a model.

Example:
python -m scripts.export_chapter4_explainability \
    --model artifacts/churniq_full_candidate.joblib \
    --data data/sample/rees46_chapter4_holdout.csv \
    --output data/reports/chapter4_explainability
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from churniq.data import load_csv, validate_prediction_data
from churniq.explain import explain
from churniq.model import load


def gain_by_selected_feature(model, selected_names: list[str]) -> pd.Series:
    """Get XGBoost *total* gain for each selected transformed feature.

    A trained model may not split on every selected feature. Omitted entries
    are zeros. XGBoost stores array-input feature identifiers as f0, f1, ...
    """
    booster = model.get_booster()
    gain = booster.get_score(importance_type="total_gain")
    result = []
    for i, name in enumerate(selected_names):
        value = gain.get(f"f{i}", gain.get(str(name), 0.0))
        result.append(float(value))
    return pd.Series(result, index=selected_names, name="xgb_total_gain")


def make_importance_table(selected_names: list[str],
                          shap_values: np.ndarray,
                          gains: pd.Series) -> pd.DataFrame:
    """Join SHAP and XGBoost importance with consistent selected-feature order."""
    values = np.asarray(shap_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(selected_names) or values.shape[0] == 0:
        raise ValueError("SHAP results do not match the selected feature dimensions.")
    if list(gains.index) != list(selected_names):
        raise ValueError("Gain values are not aligned with the selected feature names.")
    means = np.abs(values).mean(axis=0)
    total = gains.sum()
    result = pd.DataFrame({
        "feature": selected_names,
        "mean_abs_shap_log_odds": means,
        "xgb_total_gain": gains.to_numpy(),
        "xgb_total_gain_share_pct": (
            (gains.to_numpy() / total * 100) if total > 0 else np.zeros(len(gains))
        ),
    })
    result["shap_rank"] = result["mean_abs_shap_log_odds"].rank(
        method="min", ascending=False
    ).astype(int)
    result["gain_rank"] = result["xgb_total_gain"].rank(
        method="min", ascending=False
    ).astype(int)
    return result.sort_values(
        "mean_abs_shap_log_odds", ascending=False
    ).reset_index(drop=True)


def _bars(values: pd.DataFrame, name_col: str, score_col: str,
          title: str, xlabel: str, path: Path, top: int = 12):
    subset = values.nlargest(top, score_col).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(subset[name_col].str.replace("num__", "", regex=False)
            .str.replace("cat__", "", regex=False), subset[score_col], color="#2B7192")
    ax.set_title(title, fontsize=14, pad=14)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_chapter4_explainability(model_path: str | Path,
                                   data_path: str | Path,
                                   output_path: str | Path,
                                   max_rows: int = 200,
                                   top: int = 12) -> dict:
    if not 1 <= max_rows <= 300:
        raise ValueError("--max-rows must be between 1 and 300.")
    if not 1 <= top <= 40:
        raise ValueError("--top must be between 1 and 40.")

    model_path, data_path, output_path = map(
        Path, (model_path, data_path, output_path)
    )
    artifact = load(model_path)  # Load only your own trusted .joblib file.
    raw = load_csv(data_path)
    if any(str(c).startswith("target_") for c in raw.columns):
        raise ValueError(
            "Provide an unlabelled, held-out customer sample, not outcome-labelled training rows."
        )
    raw = raw.head(max_rows).copy()
    x = validate_prediction_data(raw, artifact["features"],
                                 artifact["numeric"], artifact["categorical"])
    selected_names = list(artifact["selected_names"])
    if not selected_names:
        raise ValueError("Saved model contains no selected feature names.")

    pipe = artifact["pipeline"]
    probabilities = pipe.predict_proba(x)[:, 1]
    result = explain(artifact, x, max_rows=max_rows)
    values = np.asarray(result["values"])
    if values.shape != (len(raw), len(selected_names)):
        raise ValueError("SHAP output does not match the number of selected features.")

    gains = gain_by_selected_feature(pipe.named_steps["model"], selected_names)
    importance = make_importance_table(selected_names, values, gains)
    output_path.mkdir(parents=True, exist_ok=True)
    table_path = output_path / "xgboost_shap_importance_comparison.csv"
    importance.to_csv(table_path, index=False)

    _bars(importance, "feature", "xgb_total_gain_share_pct",
          "XGBoost Importance — Total Gain", "Share of all tree-split gain (%)",
          output_path / "figure_xgboost_total_gain.png", top=top)
    _bars(importance, "feature", "mean_abs_shap_log_odds",
          "Global SHAP Importance — Held-Out Customer Sample",
          "Mean absolute SHAP contribution (log-odds)",
          output_path / "figure_shap_global_importance.png", top=top)

    # Illustrative example is chosen deterministically by score; it is not
    # representative of every customer. No customer ID appears in the file.
    row = int(np.argmax(probabilities))
    local = shap.Explanation(
        values=values[row],
        base_values=float(np.asarray(result["base_values"]).reshape(-1)[row]),
        data=np.asarray(result["feature_values"])[row],
        feature_names=selected_names,
    )
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(local, max_display=top, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"Individual SHAP Explanation — Sample Record {row + 1} "
        "(Highest Model Score)",
        y=1.03, fontsize=13,
    )
    fig.savefig(output_path / "figure_shap_local_waterfall.png",
                dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "source_model": str(model_path),
        "unlabelled_sample": str(data_path),
        "sample_records_explained": int(len(raw)),
        "selected_feature_count": int(len(selected_names)),
        "illustrative_row_number_one_based": row + 1,
        "illustrative_model_score": float(probabilities[row]),
        "model_trained_on_rows": artifact["report"].get("source_rows"),
        "model_used": artifact["report"].get("model_used"),
        "model_trained_at_utc": artifact["report"].get("trained_at_utc"),
        "shap_units": "log-odds of the model's four-week non-purchase output",
        "gain_definition": "XGBoost total_gain normalised to 100% over selected features",
        "limitations": (
            "Explainability is model attribution, not causation. "
            "Sample reflects provided historical records, not independent future deployment. "
            "Uncalibrated scores must not be interpreted as exact customer probabilities. "
            "Feature provenance and time-window leakage require independent review."
        ),
    }
    (output_path / "explainability_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Create Chapter 4 XGBoost gain and SHAP figures without retraining."
    )
    parser.add_argument("--model", default="artifacts/churniq_full_candidate.joblib")
    parser.add_argument("--data", default="data/sample/rees46_chapter4_holdout.csv")
    parser.add_argument("--output", default="data/reports/chapter4_explainability")
    parser.add_argument("--max-rows", type=int, default=200)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    meta = export_chapter4_explainability(
        args.model, args.data, args.output, args.max_rows, args.top
    )
    print(f"Generated Chapter 4 importance table and 3 PNG figures in {args.output}.")
    print(f"Explained {meta['sample_records_explained']} historical, unlabelled records.")
    print("The trained model and REES46 dataset were not modified.")


if __name__ == "__main__":
    main()
