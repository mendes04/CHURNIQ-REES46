"""Regression tests for standard e-commerce order CSV summary.

These fixtures are fictional and do not use the private REES46 dataset.
"""
import pandas as pd
import pytest

from churniq.store_orders import summarize_orders, suggest_order_columns


def sample_orders():
    return pd.DataFrame({
        "Customer ID": ["C1", "C1", "C1", "C2", "C3"],
        "Name": ["O1", "O1", "O2", "O3", "O4"],
        "Created at": [
            "2026-10-01", "2026-10-01", "2026-10-20",
            "2026-09-01", "2026-10-19",
        ],
        "Total": ["RM 120.00", "RM 120.00", "60.00", "75.00", "55.00"],
    })


def test_column_suggestions_and_deduplicated_order_totals():
    df = sample_orders()
    assert suggest_order_columns(df.columns) == {
        "customer": "Customer ID",
        "order": "Name",
        "date": "Created at",
        "amount": "Total",
    }
    report, meta = summarize_orders(
        df, customer_col="Customer ID", order_col="Name",
        date_col="Created at", amount_col="Total",
        reference_date="2026-10-21", inactive_days=30,
    )
    assert meta["record_rows"] == 5
    assert meta["orders"] == 4
    assert meta["customers"] == 3
    assert meta["customers_without_recent_order"] == 1
    c1 = report.set_index("Customer ID").loc["C1"]
    assert c1["Orders"] == 2
    assert c1["Total order value"] == pytest.approx(180.0)
    assert c1["Days since last order"] == 1


def test_line_items_are_summed_within_order():
    df = sample_orders()
    df.loc[1, "Total"] = "40.00"
    report, _ = summarize_orders(
        df, customer_col="Customer ID", order_col="Name",
        date_col="Created at", amount_col="Total",
        amount_type="line_total", reference_date="2026-10-21"
    )
    assert report.set_index("Customer ID").loc["C1", "Total order value"] == pytest.approx(220.0)


def test_conflicting_repeated_order_totals_are_rejected():
    df = sample_orders()
    df.loc[1, "Total"] = "40.00"
    with pytest.raises(ValueError, match="conflicting"):
        summarize_orders(
            df, customer_col="Customer ID", order_col="Name",
            date_col="Created at", amount_col="Total",
        )


def test_missing_date_and_earlier_asof_date_rejected():
    df = sample_orders()
    df.loc[2, "Created at"] = "not a date"
    with pytest.raises(ValueError, match="dates"):
        summarize_orders(df, customer_col="Customer ID", date_col="Created at")
    with pytest.raises(ValueError, match="before some orders"):
        summarize_orders(sample_orders(), customer_col="Customer ID", date_col="Created at",
                         reference_date="2026-09-10")


def test_no_amount_is_valid_and_each_row_counts_without_order_id():
    df = sample_orders()
    report, meta = summarize_orders(
        df, customer_col="Customer ID", date_col="Created at",
        reference_date="2026-10-21"
    )
    assert meta["orders"] == 5
    assert meta["no_order_id"] is True
    assert "Total order value" not in report
    assert len(report) == 3
