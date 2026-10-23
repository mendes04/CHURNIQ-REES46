"""Plain-language CHURNIQ presentation tests."""
import pandas as pd
import pytest

from churniq.business import (
    display_predictions, display_priority, readable_effect,
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
    assert readable.iloc[0]["Risk score (not calibrated)"] == pytest.approx(85.0)
    assert readable.iloc[1]["Priority"] == "Lower priority"
    assert readable.iloc[1]["Model's 4-week prediction"] == "Purchase expected"
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
