"""Descriptive analysis of ordinary e-commerce order exports.

This module does not use the REES46-trained churn model or assign churn
probabilities. It expects paid/completed order data supplied by the store.
"""
from __future__ import annotations

import re
from datetime import date, datetime

import numpy as np
import pandas as pd


ALIASES = {
    "customer": (
        "customer_id", "customerid", "customer_email", "email", "billing_email",
        "buyer_id", "client_id", "user_id", "customer", "contact_email",
    ),
    "order": ("order_id", "order_number", "order_no", "order_name", "name", "id"),
    "date": (
        "order_date", "created_at", "created_at_utc", "created", "processed_at",
        "purchase_date", "date", "order_time", "timestamp",
    ),
    "amount": (
        "order_total", "total_price", "total", "total_sales", "grand_total",
        "total_amount", "amount", "line_item_total", "line_total", "subtotal",
    ),
}


def suggest_order_columns(columns) -> dict[str, str | None]:
    norm = {re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_"): c for c in columns}
    return {key: next((norm[v] for v in aliases if v in norm), None)
            for key, aliases in ALIASES.items()}


def _currency_values(series: pd.Series) -> pd.Series:
    # Supports usual decimal format, e.g. $1,240.55, RM 1,240.55.
    cleaned = (series.astype("string").str.strip()
               .str.replace(r"[A-Za-z$£€₹\s]", "", regex=True)
               .str.replace(",", "", regex=False))
    values = pd.to_numeric(cleaned, errors="coerce")
    if values.isna().any() or not np.isfinite(values.astype(float)).all():
        raise ValueError("The amount column includes missing/non-numeric values. "
                         "Export a numeric order total (e.g. 120.50) or choose 'No amount'.")
    return values.astype(float)


def summarize_orders(
    data: pd.DataFrame,
    *,
    customer_col: str,
    date_col: str,
    order_col: str | None = None,
    amount_col: str | None = None,
    amount_type: str = "order_total",
    reference_date: date | datetime | str | None = None,
    inactive_days: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate transaction rows without creating predictive churn scores.

    'order_total': repeated order totals are counted once per order.
    'line_total': line amounts are summed per order.
    No order ID: each row is treated as a separate order.
    All statuses are included, so filter cancelled/refunded rows upstream.
    """
    if data.empty:
        raise ValueError("The uploaded file is empty.")
    if inactive_days < 1:
        raise ValueError("The lookback period must be at least one day.")
    if amount_type not in {"order_total", "line_total"}:
        raise ValueError("Choose order_total or line_total for amount handling.")
    for col in (customer_col, date_col, order_col, amount_col):
        if col is not None and col not in data:
            raise ValueError(f"Column not present in uploaded CSV: {col}")
    if customer_col == date_col or (order_col is not None and order_col in {customer_col, date_col}):
        raise ValueError("Customer, order and date must use different columns.")

    customers = data[customer_col].astype("string").str.strip()
    if customers.isna().any() or customers.eq("").any():
        raise ValueError("Some order rows have no customer identifier. Fix or filter these records first.")
    timestamps = pd.to_datetime(data[date_col], errors="coerce", utc=True, format="mixed")
    if timestamps.isna().any():
        raise ValueError("Some order dates could not be understood. Use an ISO date such as 2026-01-15.")

    if order_col is not None:
        orders = data[order_col].astype("string").str.strip()
        if orders.isna().any() or orders.eq("").any():
            raise ValueError("Some rows have no order ID. Fix these before analysing orders.")
    else:
        orders = pd.Series([f"ROW-{i}" for i in range(len(data))], index=data.index)

    frame = pd.DataFrame({"_customer": customers.to_numpy(),
                          "_order": orders.to_numpy(),
                          "_date": timestamps.to_numpy()}, index=data.index)
    if amount_col is not None:
        frame["_amount"] = _currency_values(data[amount_col]).to_numpy()
        if amount_type == "order_total":
            inconsistent = (frame.groupby(["_customer", "_order"])["_amount"].nunique() > 1)
            if inconsistent.any():
                raise ValueError("The same order ID has conflicting order-total values. "
                                 "Choose line-item amounts if these values belong to different items.")

    group = frame.groupby(["_customer", "_order"], sort=False, dropna=False)
    aggregation = {"_date": "max"}
    if amount_col is not None:
        aggregation["_amount"] = "first" if amount_type == "order_total" else "sum"
    per_order = group.agg(aggregation).reset_index()

    latest = per_order["_date"].max()
    if reference_date is None:
        reference = latest.normalize()
    else:
        parsed = pd.Timestamp(reference_date)
        reference = (parsed.tz_localize("UTC") if parsed.tzinfo is None
                     else parsed.tz_convert("UTC")).normalize()
    if (per_order["_date"].dt.normalize() > reference).any():
        raise ValueError("The chosen 'as of' date is before some orders. Choose a later date.")

    customer_group = per_order.groupby("_customer", sort=False, dropna=False)
    summary = customer_group.agg(
        Orders=("_order", "nunique"),
        **{"Last order": ("_date", "max"), "First order": ("_date", "min")},
    ).reset_index().rename(columns={"_customer": "Customer ID"})
    summary["Days since last order"] = (
        reference - summary["Last order"].dt.normalize()
    ).dt.days.astype(int)
    summary["Order activity"] = np.where(
        summary["Days since last order"] >= inactive_days,
        f"No order in {inactive_days}+ days", "Ordered recently"
    )
    if amount_col is not None:
        spending = customer_group["_amount"].sum().rename("Total order value").reset_index()
        spending = spending.rename(columns={"_customer": "Customer ID"})
        summary = summary.merge(spending, on="Customer ID", how="left", validate="one_to_one")
        summary["Average order value"] = summary["Total order value"] / summary["Orders"]

    summary["First order"] = summary["First order"].dt.date.astype(str)
    summary["Last order"] = summary["Last order"].dt.date.astype(str)
    summary = summary.sort_values(["Days since last order", "Orders"], ascending=[False, False])
    summary = summary.reset_index(drop=True)
    metadata = {
        "record_rows": int(len(frame)),
        "orders": int(len(per_order)),
        "customers": int(len(summary)),
        "latest_order_date": str(latest.date()),
        "reference_date": str(reference.date()),
        "inactive_days": int(inactive_days),
        "customers_without_recent_order": int((summary["Days since last order"] >= inactive_days).sum()),
        "has_amount": amount_col is not None,
        "no_order_id": order_col is None,
        "all_statuses_included": True,
    }
    return summary, metadata
