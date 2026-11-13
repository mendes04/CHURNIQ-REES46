"""Plain-language presentation helpers for CHURNIQ's e-commerce audience.

These helpers never alter prediction probabilities or model features.
They rename display labels only, keeping scientific names available for audits.
"""
from __future__ import annotations

import re

import pandas as pd


FEATURE_LABELS = {
    "session_recency_min": "Recent shopping visits",
    "session_recency_mean": "Typical time since a shopping visit",
    "session_count_month_lag0": "Shopping visits in the latest month",
    "session_count_month_lag1": "Shopping visits in the previous month",
    "session_count_month_ma3": "Average monthly shopping visits",
    "session_count_month_mean": "Typical number of monthly visits",
    "purchase_recency_min": "Recent purchasing activity",
    "purchase_recency_mean": "Typical time since a purchase",
    "purchase_number_mean": "Average purchase frequency",
    "purchase_number_cv": "Changes in purchase frequency",
    "purchase_number_sum": "Total purchases recorded",
    "haspurchase_max": "Previous purchasing activity",
    "session_number_mean": "Average shopping sessions",
    "session_number_cv": "Changes in shopping sessions",
    "start_monthday_max": "Calendar timing of visits",
    "revenue_sum": "Amount spent with the store",
    "purchase_revenue_sum": "Previous purchase value",
    "session_count_month": "Monthly shopping visits",
}

FAMILY_LABELS = (
    ("session", "Shopping visits"),
    ("purchase", "Purchase history"),
    ("revenue", "Customer spending"),
    ("cart", "Shopping cart activity"),
    ("product", "Product browsing"),
    ("item", "Products viewed"),
    ("category", "Product categories explored"),
    ("recency", "Time since last activity"),
    ("frequency", "Shopping frequency"),
    ("interaction", "Store interactions"),
)


def readable_feature(feature: str) -> str:
    """Return a conservative, readable label, with technical name still accessible."""
    raw = str(feature).replace("num__", "", 1).replace("cat__", "", 1)
    if raw in FEATURE_LABELS:
        return FEATURE_LABELS[raw]
    for fragment, family in FAMILY_LABELS:
        if fragment in raw.lower():
            suffix = []
            low = raw.lower()
            if "mean" in low or "avg" in low or "ma3" in low:
                suffix.append("average")
            elif "cv" in low or "_sd" in low:
                suffix.append("variation")
            elif "lag" in low:
                suffix.append("previous period")
            elif "min" in low:
                suffix.append("minimum")
            elif "max" in low:
                suffix.append("maximum")
            return family + (f" ({', '.join(suffix)})" if suffix else "")
    return re.sub(r"[_]+", " ", raw).strip().capitalize()


def readable_effect(contribution: float) -> str:
    """SHAP log-odds are directional, not causal."""
    if contribution > 0:
        return "Pushed risk estimate up"
    if contribution < 0:
        return "Pushed risk estimate down"
    return "Little effect on this estimate"


def display_priority(raw_band: str) -> str:
    return {
        "High": "Needs attention",
        "Moderate": "Keep an eye on",
        "Low": "Lower priority",
    }.get(str(raw_band), "Uncategorised")


def display_predictions(result: pd.DataFrame) -> pd.DataFrame:
    """Display-facing copy of predictions; original scores remain unchanged."""
    needed = {
        "customer_id", "predicted_no_purchase_risk",
        "predicted_no_purchase_next_4_weeks", "risk_band"
    }
    missing = needed.difference(result.columns)
    if missing:
        raise ValueError(f"Prediction results missing: {sorted(missing)}")
    output = pd.DataFrame({
        "Customer ID": result["customer_id"].astype(str),
        "Priority": result["risk_band"].map(display_priority),
        "Churn risk score (not calibrated)": (
            result["predicted_no_purchase_risk"].astype(float) * 100
        ).round(1),
        "Predicted churn status (proxy)": result[
            "predicted_no_purchase_next_4_weeks"
        ].map({
            1: "Churn risk flagged",
            0: "Not flagged for churn",
        }),
    })
    return output


def suggested_review_action(band: str) -> str:
    """One short, advisory action per review-priority band."""
    return {
        "High": "Check recent orders and support issues; consider a helpful follow-up.",
        "Moderate": "Watch activity and share relevant updates where consent allows.",
        "Low": "Continue good service and monitor for changes.",
    }.get(str(band), "Review the customer's history first.")


def recommended_actions_for_customers(result: pd.DataFrame) -> pd.DataFrame:
    """Add practical review ideas without changing predictions or scores."""
    view = display_predictions(result)
    view["Suggested next step"] = result["risk_band"].map(suggested_review_action).values
    return view


def churn_overview(result: pd.DataFrame) -> dict:
    """Summarise *model flags*, never claim to measure observed churn.

    Count RECORDS, not unique customers. The classification flag follows the
    threshold used at prediction time, while priority bands follow model-score
    demonstration ranges and must not be confused with those flags.
    """
    if result.empty:
        return {
            "records": 0, "flagged": 0, "flagged_pct": 0.0,
            "high": 0, "moderate": 0, "low": 0, "average_score": 0.0,
        }
    required = {
        "predicted_no_purchase_next_4_weeks",
        "predicted_no_purchase_risk", "risk_band",
    }
    missing = required.difference(result.columns)
    if missing:
        raise ValueError(f"Prediction results missing: {sorted(missing)}")
    total = len(result)
    flagged = int(result["predicted_no_purchase_next_4_weeks"].eq(1).sum())
    return {
        "records": total,
        "flagged": flagged,
        "flagged_pct": 100 * flagged / total,
        "high": int(result["risk_band"].eq("High").sum()),
        "moderate": int(result["risk_band"].eq("Moderate").sum()),
        "low": int(result["risk_band"].eq("Low").sum()),
        "average_score": 100 * float(result["predicted_no_purchase_risk"].mean()),
    }


OVERALL_RETENTION_ACTIONS = (
    {
        "icon": "🔍",
        "label": "Review customers at risk",
        "description": "Check recent orders, shopping visits and support issues before acting.",
        "tone": "rose",
    },
    {
        "icon": "💬",
        "label": "Offer helpful support",
        "description": "Fix service problems and consider a useful reminder when allowed.",
        "tone": "purple",
    },
    {
        "icon": "🎁",
        "label": "Test relevant offers",
        "description": "Try suitable updates or small offers for consenting customers. Measure the results.",
        "tone": "amber",
    },
    {
        "icon": "📈",
        "label": "Keep learning",
        "description": "Track future purchases and compare outcomes before claiming an action worked.",
        "tone": "teal",
    },
)


def suggested_review_checks(shap_contributions, limit: int = 3) -> list[str]:
    """Business review prompts from positive model attributions.

    Each (feature, log_odds_contribution) is SHAP output. Positive influence
    only signals model behaviour; these prompts are not causes or treatments.
    """
    prompts = {
        "purchase": "Check the customer's recent orders and whether their buying pattern has changed.",
        "session": "Review recent shopping visits and whether browsing activity has changed.",
        "cart": "Review basket activity and whether a shopping journey was left unfinished.",
        "revenue": "Check previous spending patterns and order values.",
        "view": "Review which products the customer browsed and whether their interests have shifted.",
        "product": "Review recent product browsing and whether displayed items remain relevant.",
        "category": "Look at the product categories they usually explore.",
        "interaction": "Review the customer's recent interactions with the online store.",
    }
    ordered = sorted(shap_contributions, key=lambda pair: float(pair[1]), reverse=True)
    checks = []
    for feature, contribution in ordered:
        if float(contribution) <= 0:
            continue
        technical_name = str(feature).lower().replace("num__", "").replace("cat__", "")
        matched = next((text for key, text in prompts.items() if key in technical_name), None)
        if matched and matched not in checks:
            checks.append(matched)
        if len(checks) >= limit:
            break
    if not checks:
        checks.append("Review recent orders, visits and customer-service history before deciding on a follow-up.")
    return checks[:limit]
