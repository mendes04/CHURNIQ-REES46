"""Human-readable thesis alignment for the CHURNIQ Streamlit dashboard.

This module changes DISPLAY WORDING ONLY. It deliberately does not reinterpret
target_event, invert class labels, retrain, or modify stored model artifacts.

The thesis Chapter 3 Table 3.2 describes 1 as no customer interaction in the
subsequent month. Existing model code/report documents 1 as no purchase in the
following four weeks. These are not interchangeable outcomes. Resolving the
discrepancy requires verifying the original REES46 label construction.
"""

PROJECT_TITLE = "Explainable Customer Churn Prediction in E-Commerce Using XGBoost and SHAP"

THESIS_CHURN_DEFINITION = (
    "Thesis (Chapter 3, Table 3.2): label 1 = no customer interaction "
    "in the subsequent month; label 0 = customer returned."
)

MODEL_CHURN_DEFINITION = (
    "Current model implementation: label 1 = no purchase during the following "
    "four weeks; label 0 = a purchase occurred."
)

CHURN_PROXY_NOTE = (
    "CHURNIQ presents the saved model's output as a churn risk score. "
    "The existing training code describes the outcome as no purchase in four weeks, "
    "while the thesis describes no interaction during the following month. "
    "The exact REES46 target definition requires verification. "
    "The score is neither proof of permanent customer loss nor a calibrated "
    "real-world probability."
)

TARGET_RECONCILIATION_NOTE = (
    "Target-definition review required: the thesis defines churn as no "
    "interaction in the subsequent month, but the current Python training "
    "code documents no purchase in the following four weeks. These are "
    "different outcomes. Verify the source REES46 target_event label before "
    "claiming the code and thesis use identical churn definitions. Do not "
    "change the class wording, reconstruct labels, or report new metrics "
    "without source verification and an appropriately rerun experiment."
)


def model_artifact_label(model_path: str) -> str:
    """Explain which locally stored research model is being shown."""
    if model_path.endswith("churniq_full_candidate.joblib"):
        return "Full candidate: tuned XGBoost (thesis experiment)"
    if model_path.endswith("churniq_rees46.joblib"):
        return "Earlier demonstration model (fast training experiment)"
    return "Unknown model artifact"
