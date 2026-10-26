"""Order-export readiness and leakage-aware historical examples.

Separate from REES46 XGBoost. Uses only order counts/timestamps.
NO saved model is fit or used here.

Time convention: at a snapshot start date t (UTC), only orders before t
can influence predictors. Label 1 means no order in [t, t+28d).
Labels are retrospective and require complete future order observations.
The latest order timestamp is NOT proof that the export is complete;
merchants must verify the end-of-data/paid-orders coverage separately.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


HISTORY_DAYS = 56
HORIZON_DAYS = 28
ACTIVE_LOOKBACK_DAYS = 180


def canonical_orders(df: pd.DataFrame, *, customer_col: str,
                     date_col: str, order_col: str | None = None) -> pd.DataFrame:
    """Return one unique historical purchase per customer/order ID.

    No guessing payment statuses: callers must filter non-purchases first.
    """
    if df.empty:
        raise ValueError("Upload an order file with records first.")
    if customer_col == date_col or (
        order_col is not None and order_col in {customer_col, date_col}
    ):
        raise ValueError("Customer, date, and order ID columns must be distinct.")
    for field in (customer_col, date_col, order_col):
        if field is not None and field not in df.columns:
            raise ValueError(f"Missing source column: {field}")

    customers = df[customer_col].astype("string").str.strip()
    if customers.isna().any() or customers.eq("").any():
        raise ValueError("Some rows have missing customer IDs.")
    dates = pd.to_datetime(df[date_col], errors="coerce", utc=True, format="mixed")
    if dates.isna().any():
        raise ValueError("Some order dates are invalid.")
    if order_col is not None:
        order_ids = df[order_col].astype("string").str.strip()
        if order_ids.isna().any() or order_ids.eq("").any():
            raise ValueError("Some rows have missing order IDs.")
    else:
        order_ids = pd.Series([f"ROW-{i}" for i in range(len(df))], dtype="string")

    output = pd.DataFrame({
        "customer_id": customers.reset_index(drop=True),
        "order_id": order_ids.reset_index(drop=True),
        "order_date": dates.reset_index(drop=True),
    })
    return (
        output.groupby(["customer_id", "order_id"], as_index=False, sort=False)
        .agg(order_date=("order_date", "min"))
        .sort_values(["order_date", "customer_id", "order_id"])
        .reset_index(drop=True)
    )


def generate_order_examples(
    orders: pd.DataFrame, *, max_snapshots: int = 4,
    history_days: int = HISTORY_DAYS,
    horizon_days: int = HORIZON_DAYS,
    active_lookback_days: int = ACTIVE_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Reconstruct labelled past snapshots without future-feature leakage.

    Uses a maximum of four earliest-to-latest weekly dates by default.
    Historical records are chosen based on information known before t.
    This is demonstration/preparation, NOT a trained classifier.
    """
    if max_snapshots < 1 or history_days < 1 or horizon_days < 1:
        raise ValueError("Snapshots and observation windows must be positive.")
    if active_lookback_days < history_days:
        raise ValueError("Active customer lookback cannot be shorter than feature history.")
    if orders.empty:
        return pd.DataFrame()

    timestamps = pd.to_datetime(orders["order_date"], utc=True)
    first = timestamps.min().normalize()
    last = timestamps.max().normalize()
    earliest = first + pd.Timedelta(days=history_days)
    latest = last - pd.Timedelta(days=horizon_days)
    if earliest > latest:
        return pd.DataFrame()

    # Use weekly snapshots. Include the latest eligible snapshot if needed.
    possible = list(pd.date_range(earliest, latest, freq="7D", tz="UTC"))
    if not possible:
        return pd.DataFrame()
    if len(possible) > max_snapshots:
        picks = np.linspace(0, len(possible) - 1, num=max_snapshots, dtype=int)
        snapshot_dates = [possible[int(i)] for i in dict.fromkeys(picks)]
    else:
        snapshot_dates = possible

    values = orders.assign(order_date=timestamps)
    batches = []
    for snapshot in snapshot_dates:
        historical = values.loc[values["order_date"] < snapshot].copy()
        future = values.loc[
            (values["order_date"] >= snapshot) &
            (values["order_date"] < snapshot + pd.Timedelta(days=horizon_days))
        ]
        if historical.empty:
            continue
        grouped = historical.groupby("customer_id")["order_date"]
        last_order = grouped.max()
        first_order = grouped.min()
        age_days = (snapshot - first_order).dt.days
        recency = (snapshot - last_order).dt.days
        eligible = last_order.index[recency.le(active_lookback_days)]
        if len(eligible) == 0:
            continue

        recent_28 = historical[
            historical["order_date"] >= snapshot - pd.Timedelta(days=28)
        ].groupby("customer_id").size()
        recent_window = historical[
            historical["order_date"] >= snapshot - pd.Timedelta(days=history_days)
        ].groupby("customer_id").size()
        future_ids = set(future["customer_id"].astype(str))

        rows = pd.DataFrame({
            "customer_id": pd.Series(eligible, dtype="string").reset_index(drop=True),
            "snapshot_date": snapshot.date().isoformat(),
        })
        identifiers = rows["customer_id"]
        rows["days_since_last_order"] = identifiers.map(recency).astype(int)
        rows["customer_age_days"] = identifiers.map(age_days).astype(int)
        rows["orders_before_snapshot"] = identifiers.map(grouped.size()).astype(int)
        rows["orders_previous_28d"] = identifiers.map(recent_28).fillna(0).astype(int)
        rows["orders_previous_56d"] = identifiers.map(recent_window).fillna(0).astype(int)
        rows["target_no_purchase_next_28d"] = (
            ~identifiers.isin(future_ids)
        ).astype(int)
        batches.append(rows)
    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()


def assess_prediction_readiness(orders: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Conservative *heuristic* check, not a statistical quality guarantee.

    A real deployment still needs coverage checks, additional longitudinal
    snapshots, store-specific feature selection, validation and calibration.
    """
    if orders.empty:
        raise ValueError("No valid orders to analyse.")
    first = orders["order_date"].min().date()
    last = orders["order_date"].max().date()
    coverage = (last - first).days
    example = generate_order_examples(orders)
    counts = (
        example["target_no_purchase_next_28d"].value_counts()
        if not example.empty else pd.Series(dtype=int)
    )
    positives = int(counts.get(1, 0))
    negatives = int(counts.get(0, 0))
    distinct_customers = int(orders["customer_id"].nunique())
    unique_orders = len(orders)
    n_snapshots = int(example["snapshot_date"].nunique()) if not example.empty else 0
    requirements = {
        "At least 12 weeks of order history": coverage >= 84,
        "At least 1,000 recorded orders": unique_orders >= 1000,
        "At least 200 distinct customers": distinct_customers >= 200,
        "At least 3 historical snapshots": n_snapshots >= 3,
        "At least 50 examples of each outcome": min(positives, negatives) >= 50,
    }
    return {
        "earliest_order": str(first),
        "latest_order": str(last),
        "history_days": int(coverage),
        "unique_orders": int(unique_orders),
        "unique_customers": distinct_customers,
        "snapshot_dates": n_snapshots,
        "eligible_customer_periods": len(example),
        "examples_no_future_purchase": positives,
        "examples_with_future_purchase": negatives,
        "minimum_checks": requirements,
        "passes_basic_checks": bool(all(requirements.values())),
        "caution": (
            "These are illustrative screening checks, not sufficient conditions "
            "for a valid predictive model. A snapshot needs a COMPLETE 28-day "
            "future purchase record. The last order date alone does not prove "
            "the export is complete. This does not train or score a model."
        ),
    }, example
