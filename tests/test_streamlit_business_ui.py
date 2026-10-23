"""Smoke-test Streamlit navigation even when no private model artifact is installed.

CI must remain independent of the research dataset and user's trained joblib.
"""
from pathlib import Path

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
