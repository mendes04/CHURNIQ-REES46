"""Fictional-order tests for prediction-readiness and time-safe features."""
import pandas as pd

from churniq.order_readiness import (
    assess_prediction_readiness,
    canonical_orders,
    generate_order_examples,
)


def _events():
    return pd.DataFrame({
        "customer": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "order": ["A1", "A1", "A2", "B1", "B2", "B3", "C1", "C2", "C3"],
        "date": [
            "2026-01-01", "2026-01-01", "2026-03-04",
            "2026-01-15", "2026-02-22", "2026-03-15",
            "2026-02-01", "2026-04-01", "2026-04-30",
        ],
    })


def test_duplicate_line_items_not_counted_twice():
    output = canonical_orders(_events(), customer_col="customer",
                              date_col="date", order_col="order")
    assert len(output) == 8
    assert (output.customer_id == "A").sum() == 2


def test_no_future_transactions_in_training_features():
    orders = canonical_orders(
        _events(), customer_col="customer", date_col="date", order_col="order"
    )
    rows = generate_order_examples(orders, max_snapshots=20)
    first_snapshot = rows[rows.snapshot_date == "2026-02-26"].set_index("customer_id")
    assert set(first_snapshot.index) == {"A", "B", "C"}
    assert first_snapshot.loc["A", "orders_before_snapshot"] == 1
    assert first_snapshot.loc["A", "orders_previous_28d"] == 0
    assert first_snapshot.loc["A", "target_no_purchase_next_28d"] == 0
    assert first_snapshot.loc["B", "orders_before_snapshot"] == 2
    assert first_snapshot.loc["C", "target_no_purchase_next_28d"] == 1
    assert (rows["days_since_last_order"] >= 0).all()


def test_four_order_demo_is_not_ready_for_prediction():
    df = pd.DataFrame({
        "customer": ["C001", "C001", "C002", "C003"],
        "order": ["O100", "O101", "O102", "O103"],
        "date": ["2026-09-04", "2026-10-09", "2026-08-23", "2026-10-10"],
    })
    orders = canonical_orders(df, customer_col="customer",
                              date_col="date", order_col="order")
    report, examples = assess_prediction_readiness(orders)
    assert report["unique_orders"] == 4
    assert report["unique_customers"] == 3
    assert report["passes_basic_checks"] is False
    assert report["eligible_customer_periods"] == 0
    assert examples.empty


def test_longer_history_creates_snapshot_examples():
    orders = canonical_orders(
        _events(), customer_col="customer", date_col="date", order_col="order"
    )
    report, rows = assess_prediction_readiness(orders)
    assert report["snapshot_dates"] >= 1
    assert report["examples_no_future_purchase"] > 0
    assert report["examples_with_future_purchase"] > 0
    assert not rows.empty
    assert "customer_id" in rows and "target_no_purchase_next_28d" in rows
