"""Tests for saved-model Chapter 4 feature-importance export utilities.

Uses fictional data; does not load REES46 or the private trained artifact.
"""
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from scripts.export_chapter4_explainability import (
    gain_by_selected_feature,
    make_importance_table,
    _bars,
)


def test_gain_mapping_from_numpy_xgboost_features():
    x = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0, 0], [1, 0], [0, 1], [1, 1],
    ], dtype=float)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    clf = XGBClassifier(
        n_estimators=8, max_depth=2, tree_method="hist",
        n_jobs=1, random_state=42,
    )
    clf.fit(x, y)
    result = gain_by_selected_feature(clf, ["num__A", "num__B"])
    assert result.index.tolist() == ["num__A", "num__B"]
    assert (result >= 0).all()
    assert result.sum() > 0


def test_shap_gain_comparison_is_aligned():
    names = ["num__A", "num__B", "num__C"]
    values = np.array([[0.8, -0.1, 0.0], [0.2, -0.3, 0.0]])
    gains = pd.Series([5.0, 3.0, 0.0], index=names)
    table = make_importance_table(names, values, gains)
    assert table.iloc[0]["feature"] == "num__A"
    assert table.iloc[0]["mean_abs_shap_log_odds"] == pytest.approx(0.5)
    assert table.iloc[0]["xgb_total_gain_share_pct"] == pytest.approx(62.5)
    assert table["xgb_total_gain_share_pct"].sum() == pytest.approx(100.0)
    assert table["shap_rank"].tolist() == [1, 2, 3]


def test_dimension_mismatch_rejected():
    names = ["num__A", "num__B"]
    with pytest.raises(ValueError, match="dimensions"):
        make_importance_table(
            names, np.ones((4, 1)), pd.Series([1.0, 2.0], index=names)
        )


def test_importance_png_generated(tmp_path):
    ranking = pd.DataFrame({
        "feature": ["num__purchase_count", "num__session_recency"],
        "mean_abs_shap_log_odds": [0.5, 0.2],
    })
    result = tmp_path / "plot.png"
    _bars(ranking, "feature", "mean_abs_shap_log_odds",
          "SHAP test", "Mean absolute log-odds", result, top=2)
    assert result.exists()
    assert result.stat().st_size > 1000
