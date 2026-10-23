"""Plain-language CHURNIQ presentation tests."""
import pandas as pd
import pytest

from churniq.business import (
    display_predictions, display_priority, readable_effect,
    readable_feature, suggested_review_action,
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
