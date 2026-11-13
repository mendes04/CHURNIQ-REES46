"""Verify thesis-facing wording does not hide the target-definition discrepancy."""
from churniq.thesis_alignment import (
    PROJECT_TITLE,
    THESIS_CHURN_DEFINITION,
    MODEL_CHURN_DEFINITION,
    CHURN_PROXY_NOTE,
    TARGET_RECONCILIATION_NOTE,
    model_artifact_label,
)


def test_thesis_title_and_model_target_are_explicit():
    assert "Customer Churn Prediction" in PROJECT_TITLE
    assert "no customer interaction" in THESIS_CHURN_DEFINITION
    assert "no purchase" in MODEL_CHURN_DEFINITION
    assert "no purchase" in CHURN_PROXY_NOTE
    assert "no interaction" in CHURN_PROXY_NOTE
    assert "Verify" in TARGET_RECONCILIATION_NOTE


def test_full_model_is_identified_separately_from_smoke_model():
    full = model_artifact_label("artifacts/churniq_full_candidate.joblib")
    early = model_artifact_label("artifacts/churniq_rees46.joblib")
    assert "Full candidate" in full
    assert "demonstration" in early
    assert full != early
