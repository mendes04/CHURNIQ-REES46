"""Smoke-test Streamlit navigation even when no private model artifact is installed.

CI must remain independent of the research dataset and user's trained joblib.
"""
from pathlib import Path

import pandas as pd

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_business_pages_render_without_private_model():
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert len(app.exception) == 0

    navigation = app.sidebar.radio[0]
    pages = [
        "Business Home", "Check Customers", "Customer List",
        "Why Customers Are Flagged", "Follow-up Planner",
        "Research Lab", "Settings",
    ]
    for page in pages:
        app.sidebar.radio[0].set_value(page).run()
        assert len(app.exception) == 0, f"Streamlit page failed: {page}"

def test_business_home_with_realistic_prediction_state():
    """Catch Streamlit magic-expression errors only reached after uploading data."""
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    app.session_state["model_bundle"] = {
        "report": {"model_used": "Tuned XGBoost", "source_rows": 3},
        "features": ["session_count"],
    }
    app.session_state["predictions"] = pd.DataFrame({
        "customer_id": ["A", "B", "C"],
        "predicted_no_purchase_risk": [0.85, 0.56, 0.12],
        "predicted_no_purchase_risk_pct": [85.0, 56.0, 12.0],
        "predicted_no_purchase_next_4_weeks": [1, 1, 0],
        "risk_band": ["High", "Moderate", "Low"],
    })
    app.session_state["customers"] = pd.DataFrame({"session_count": [3, 5, 10]})
    app.session_state["prediction_source"] = "Test customer cohort"
    app.sidebar.radio[0].set_value("Business Home").run()
    assert len(app.exception) == 0


def test_customer_list_with_predictions_and_drilldown_control():
    """The list page must be usable after customer predictions are loaded."""
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    app.session_state["model_bundle"] = {
        "report": {"model_used": "Tuned XGBoost", "source_rows": 3},
        "features": ["session_count"],
    }
    app.session_state["predictions"] = pd.DataFrame({
        "customer_id": ["A", "B", "C"],
        "predicted_no_purchase_risk": [0.85, 0.56, 0.12],
        "predicted_no_purchase_risk_pct": [85.0, 56.0, 12.0],
        "predicted_no_purchase_next_4_weeks": [1, 1, 0],
        "risk_band": ["High", "Moderate", "Low"],
    })
    app.session_state["customers"] = pd.DataFrame({"session_count": [3, 5, 10]})
    app.sidebar.radio[0].set_value("Customer List").run()
    assert len(app.exception) == 0
    assert any(button.label.startswith("Show why") for button in app.button)
