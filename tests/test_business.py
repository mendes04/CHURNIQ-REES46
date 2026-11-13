"""Plain-language CHURNIQ presentation tests."""
import pandas as pd
import pytest

from churniq.business import (
    display_predictions, display_priority, churn_overview, recommended_actions_for_customers,
    OVERALL_RETENTION_ACTIONS, readable_effect,
    readable_feature, suggested_review_action, suggested_review_checks,
)


def test_business_customer_table_preserves_risk_without_using_jargon():
    predictions = pd.DataFrame({
        "customer_id": ["1001", "2002"],
        "predicted_no_purchase_risk": [0.85, 0.25],
        "predicted_no_purchase_next_4_weeks": [1, 0],
        "risk_band": ["High", "Low"],
    })
    readable = display_predictions(predictions)
    assert readable.iloc[0]["Customer ID"] == "1001"
    assert readable.iloc[0]["Priority"] == "Needs attention"
    assert readable.iloc[0]["Risk score"] == pytest.approx(85.0)
    assert readable.iloc[0]["Churn flag"] == "Flagged"
    assert readable.iloc[1]["Priority"] == "Lower priority"
    assert readable.iloc[1]["Churn flag"] == "Not flagged"
    assert predictions.iloc[0]["predicted_no_purchase_risk"] == 0.85


def test_nontechnical_shap_signal_labels():
    assert readable_feature("num__purchase_number_mean") == "Average purchase frequency"
    assert readable_feature("num__session_recency_min") == "Recent shopping visits"
    assert readable_effect(0.6) == "Pushed risk estimate up"
    assert readable_effect(-0.3) == "Pushed risk estimate down"
    assert "caused" not in suggested_review_action("High").lower()
    assert display_priority("Moderate") == "Keep an eye on"


def test_missing_prediction_field_rejected():
    with pytest.raises(ValueError, match="missing"):
        display_predictions(pd.DataFrame({"customer_id": ["1"]}))


def test_shap_driven_review_focus_does_not_invent_causes():
    checks = suggested_review_checks([
        ("num__session_recency_min", 2.2),
        ("num__purchase_number_mean", 0.7),
        ("num__session_count_month_lag0", 0.3),
        ("num__cart_count_month_lag0", -1.4),
    ])
    assert len(checks) == 2
    assert "shopping visits" in checks[0].lower()
    assert "orders" in checks[1].lower()
    assert not any("caused" in check.lower() for check in checks)
    assert suggested_review_checks([("num__purchase_number_mean", -0.2)])


def test_churn_percent_is_model_flag_share_not_mean_risk():
    predictions = pd.DataFrame({
        "customer_id": ["A", "B", "C", "D"],
        "predicted_no_purchase_risk": [0.9, 0.6, 0.3, 0.1],
        "predicted_no_purchase_next_4_weeks": [1, 1, 0, 0],
        "risk_band": ["High", "Moderate", "Low", "Low"],
    })
    summary = churn_overview(predictions)
    assert summary["records"] == 4
    assert summary["flagged"] == 2
    assert summary["flagged_pct"] == pytest.approx(50.0)
    assert summary["average_score"] == pytest.approx(47.5)
    assert summary["high"] == 1
    assert summary["moderate"] == 1

    # A different yes/no threshold can change flags without changing score bands.
    predictions.loc[1, "predicted_no_purchase_next_4_weeks"] = 0
    changed = churn_overview(predictions)
    assert changed["flagged_pct"] == pytest.approx(25.0)
    assert changed["high"] == 1
    assert changed["moderate"] == 1


def test_action_suggestions_are_visible_without_changing_predictions():
    source = pd.DataFrame({
        "customer_id": ["A", "B"],
        "predicted_no_purchase_risk": [0.92, 0.51],
        "predicted_no_purchase_next_4_weeks": [1, 1],
        "risk_band": ["High", "Moderate"],
    })
    before = source.copy(deep=True)
    recommended = recommended_actions_for_customers(source)
    assert "Suggested next step" in recommended.columns
    assert "support" in recommended.iloc[0]["Suggested next step"]
    assert "updates" in recommended.iloc[1]["Suggested next step"]
    pd.testing.assert_frame_equal(source, before)
    assert len(OVERALL_RETENTION_ACTIONS) >= 3
    assert any("Track" in x["description"] for x in OVERALL_RETENTION_ACTIONS)


def test_empty_churn_overview_does_not_invent_a_churn_rate():
    assert churn_overview(pd.DataFrame())["flagged_pct"] == 0.0
