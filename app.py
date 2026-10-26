"""CHURNIQ: readable customer-retention workspace with separate research tools.

Audience: e-commerce owners and retention teams, not only machine-learning researchers.
Predictions use the same saved, uncalibrated model as the existing research prototype.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from churniq.business import (
    display_predictions,
    display_priority,
    readable_effect,
    readable_feature,
    suggested_review_action,
    suggested_review_checks,
)
from churniq.data import load_csv
from churniq.explain import explain, top_global, top_local
from churniq.model import load, predict
from churniq.store_orders import suggest_order_columns, summarize_orders
from churniq.order_readiness import canonical_orders, assess_prediction_readiness


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "artifacts" / "churniq_rees46.joblib"
SAMPLE_PATH = ROOT / "data" / "sample" / "rees46_heldout_cohort.csv"
CALIBRATION_PATH = ROOT / "data" / "reports" / "rees46_calibration_full.json"
TEMPORAL_PATH = ROOT / "data" / "reports" / "rees46_temporal_full.json"

st.set_page_config(page_title="CHURNIQ | Customer Retention Workspace", page_icon="🛍️", layout="wide")
st.markdown(
    """
    <style>
      [data-testid="stApp"] {background:#0b1020;color:#edf2ff;}
      [data-testid="stSidebar"] {background:#121a2d;}
      .block-container {padding-top:2.1rem;max-width:1390px}
      h1,h2,h3 {letter-spacing:-.035em}
      [data-testid="stMetric"] {background:#172139;border:1px solid #2d3c58;
        border-radius:15px;padding:16px 18px}
      .stButton>button[kind="primary"] {background:#7868f8;color:#fff;
        border:0;border-radius:10px;font-weight:650}
      .hero {padding:25px 29px;border-radius:19px;background:
        linear-gradient(120deg,#25235d 0%,#192845 61%,#133b45 100%);
        border:1px solid #415075;margin-bottom:20px}
      .hero h1 {margin:7px 0;font-size:2.15rem;color:#fff;}
      .hero p {margin:8px 0 0;color:#d5dcf2;font-size:1.01rem;}
      .eyebrow {font-size:.76rem;font-weight:700;letter-spacing:.14em;color:#8ee4d1;}
      [data-testid="stDataFrame"] {border-radius:12px}
    </style>
    """,
    unsafe_allow_html=True,
)


def model_bundle():
    if "model_bundle" not in st.session_state:
        try:
            st.session_state.model_bundle = load(MODEL_PATH) if MODEL_PATH.exists() else None
        except Exception as error:
            st.session_state.model_bundle = None
            st.error(f"The saved prediction model could not be opened: {error}")
    return st.session_state.model_bundle


def session_predictions():
    return st.session_state.get("predictions"), st.session_state.get("customers")


def predictions_or_help():
    predictions, customers = session_predictions()
    if predictions is None:
        st.info(
            "No customers have been analysed in this session. "
            "Open **Check Customers** in the sidebar, upload a prepared file, and run the prediction."
        )
    return predictions, customers


def risk_note():
    st.caption(
        "Risk scores show the current model's estimates for no purchase in the next four weeks. "
        "They are not yet calibrated as exact real-world probabilities and do not prove a customer will leave."
    )


def title_block(eyebrow, title, description):
    st.markdown(
        '<div class="hero"><div class="eyebrow">' + eyebrow + '</div>'
        '<h1>' + title + '</h1><p>' + description + '</p></div>',
        unsafe_allow_html=True,
    )


def show_customer_table(result, limit=None):
    view = display_predictions(result)
    view = view.sort_values("Risk score (not calibrated)", ascending=False)
    st.dataframe(view.head(limit) if limit else view, use_container_width=True, hide_index=True)
    return view


def go_to(page_name, customer_index=None):
    st.session_state["navigation_page"] = page_name
    if customer_index is not None:
        st.session_state["focused_customer_idx"] = int(customer_index)


def run_customer_predictions(data, source_label, identifier=None):
    missing = [field for field in bundle["features"] if field not in data.columns]
    if missing:
        st.error(
            f"This file is missing {len(missing)} customer-behaviour fields required by "
            "the current model. A simple order export is not supported yet. "
            "Please use the prepared REES46 example or the same feature format used in training."
        )
        with st.expander("View missing field names (technical)"):
            st.write(missing)
        return False
    try:
        result = predict(
            bundle, data,
            threshold=float(st.session_state.get("decision_threshold", 0.5)),
            id_column=identifier,
        )
        st.session_state.customers = data.reset_index(drop=True)
        st.session_state.predictions = result
        st.session_state.prediction_source = source_label
        return True
    except Exception as error:
        st.error(f"Unable to analyse this file: {error}")
        return False


bundle = model_bundle()
st.sidebar.title("◈ CHURNIQ")
st.sidebar.caption("CUSTOMER RETENTION WORKSPACE")
st.sidebar.markdown("**For online stores**")
page = st.sidebar.radio(
    "Go to",
    [
        "Business Home",
        "Store Orders (Preview)",
        "Check Customers",
        "Customer List",
        "Why Customers Are Flagged",
        "Follow-up Planner",
        "Research Lab",
        "Settings",
    ],
    key="navigation_page",
)
if bundle is not None:
    st.sidebar.success("Prediction model ready")
else:
    st.sidebar.warning("Prediction model not loaded")
st.sidebar.caption("4-week non-purchase estimates · Research prototype")
st.sidebar.caption("Research Lab contains the advanced model metrics.")


if page == "Business Home":
    title_block(
        "CUSTOMER RETENTION",
        "Know who might stop buying",
        "See which customers may need attention over the next four weeks, "
        "understand the model's signals, and plan a thoughtful follow-up.",
    )
    if bundle is None:
        st.warning("Your prediction model is not loaded. Load the locally trained model before analysing customers.")
    else:
        result, _ = session_predictions()
        if result is None:
            st.subheader("Get started in three steps")
            a, b, c = st.columns(3)
            with a:
                st.markdown("#### 1. Prepare customer data")
                st.write("Use **Store Orders** for a normal orders CSV, or prepare the activity features required for model predictions.")
            with b:
                st.markdown("#### 2. Check customer risk")
                st.write("Open **Check Customers** and let CHURNIQ analyse the records.")
            with c:
                st.markdown("#### 3. Plan your next action")
                st.write("Review priority customers, see important signals, and export a review list.")
            st.info(
                "For a quick demonstration, the prepared 100-customer REES46 sample can be "
                "loaded on the Check Customers page if that file exists on this computer."
            )
        else:
            total = len(result)
            unique_customers = result["customer_id"].nunique()
            high = int((result["risk_band"] == "High").sum())
            watch = int((result["risk_band"] == "Moderate").sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Records reviewed", f"{total:,}", help="One record represents one customer at an observation period.")
            c2.metric("Needs attention", f"{high:,}", help="Current model score at least 70 out of 100.")
            c3.metric("Keep an eye on", f"{watch:,}", help="Current model score from 40 up to 70.")
            c4.metric("Average risk score", f"{result['predicted_no_purchase_risk'].mean():.0%}", help="A model score, not a calibrated probability.")
            if unique_customers != total:
                st.caption(f"These {total:,} records represent {unique_customers:,} different customers.")
            if high:
                st.success(
                    f"{high:,} of {total:,} customer records are in the highest review-priority group. "
                    "Review their history before contacting anyone."
                )
            else:
                st.info("No customers are in the highest review-priority group in this file.")
            left, right = st.columns([1, 1.3])
            with left:
                st.subheader("Where to focus")
                counts = result["risk_band"].value_counts()
                counts = pd.DataFrame({
                    "Priority": ["Needs attention", "Keep an eye on", "Lower priority"],
                    "Customer records": [
                        int(counts.get("High", 0)),
                        int(counts.get("Moderate", 0)),
                        int(counts.get("Low", 0)),
                    ],
                })
                fig = px.bar(
                    counts, x="Priority", y="Customer records",
                    color="Priority",
                    color_discrete_map={
                        "Needs attention": "#ff8193",
                        "Keep an eye on": "#f8c46b",
                        "Lower priority": "#3ad5b5",
                    },
                    text="Customer records", template="plotly_dark",
                )
                fig.update_layout(showlegend=False, xaxis_title=None, margin=dict(t=20, b=15))
                st.plotly_chart(fig, use_container_width=True)
            with right:
                st.subheader("Customers to review first")
                show_customer_table(result, limit=8)
            st.button(
                "Review the priority customer list →",
                type="primary",
                on_click=go_to, args=("Customer List",),
                key="home_customer_list",
            )
            st.caption(f"Customer data source: {st.session_state.get('prediction_source', 'This session')}")
            risk_note()



elif page == "Store Orders (Preview)":
    title_block(
        "YOUR STORE'S ORDER HISTORY",
        "Understand your customers from an orders CSV",
        "Upload ordinary transaction data to see repeat buyers, spending and customers "
        "who have not placed an order recently. No AI setup or feature engineering needed for this view.",
    )
    st.info(
        "**Order-history insights only.** This page does not use CHURNIQ's "
        "four-week prediction model, calculate customer churn probabilities, "
        "or predict whether someone will purchase again."
    )
    example_csv = (
        "customer_id,order_id,order_date,order_total\n"
        "C001,O100,2026-09-04,120.50\n"
        "C001,O101,2026-10-09,74.00\n"
        "C002,O102,2026-08-23,39.90\n"
        "C003,O103,2026-10-10,54.00\n"
    )
    st.download_button(
        "Download sample order CSV",
        data=example_csv,
        file_name="churniq_example_orders.csv",
        mime="text/csv",
    )
    orders_upload = st.file_uploader(
        "Upload orders CSV (one row per order or one row per item)",
        type=["csv"], key="store_orders_upload",
    )
    st.caption(
        "Use a paid/completed order export. Cancelled, refunded or test orders "
        "are not filtered automatically. Data stays in this Streamlit session."
    )
    if orders_upload is not None:
        try:
            orders_df = load_csv(orders_upload)
            suggestions = suggest_order_columns(orders_df.columns)
            column_names = orders_df.columns.tolist()
            st.write(f"File: **{orders_upload.name}** · {len(orders_df):,} transaction rows")
            with st.expander("Preview the imported columns"):
                st.dataframe(orders_df.head(5), hide_index=True, use_container_width=True)

            def column_index(field):
                value = suggestions.get(field)
                return column_names.index(value) if value in column_names else 0

            with st.form("analyse_store_orders"):
                st.markdown("#### Match the columns in your file")
                st.caption("Most fields are detected automatically. Please confirm them before analysing.")
                left, right = st.columns(2)
                with left:
                    customer_col = st.selectbox(
                        "Customer identifier", column_names,
                        index=column_index("customer"),
                        help="A customer ID or customer email; required to group orders by person.",
                    )
                    order_options = ["Each row is an individual order"] + column_names
                    order_guess = suggestions.get("order")
                    order_col_choice = st.selectbox(
                        "Order number or ID", order_options,
                        index=order_options.index(order_guess) if order_guess in order_options else 0,
                        help="Important when a single order has multiple item rows. "
                        "Without an order ID, each row is treated as a separate order.",
                    )
                with right:
                    date_col = st.selectbox(
                        "Date the order was placed", column_names,
                        index=column_index("date"),
                    )
                    amount_options = ["Ignore order amounts"] + column_names
                    amount_guess = suggestions.get("amount")
                    amount_col_choice = st.selectbox(
                        "Order value (optional)", amount_options,
                        index=amount_options.index(amount_guess) if amount_guess in amount_options else 0,
                    )
                amount_kind = st.radio(
                    "How is order value recorded?",
                    ["Order total (may repeat for each item)", "One amount per line item (add together)"],
                    horizontal=True,
                    help="For Shopify-style exports where order Total repeats across lines, "
                    "choose the first option. For line-item subtotals, choose the second.",
                )
                inactivity_days = st.slider(
                    "Count customers with no orders for at least this many days",
                    7, 180, 30, step=1,
                    help="This is a descriptive filter, not a prediction of future behaviour.",
                )
                try:
                    parsed_dates = pd.to_datetime(
                        orders_df[date_col], errors="coerce", utc=True, format="mixed"
                    )
                    latest_order_date = parsed_dates.max().date() if parsed_dates.notna().any() else None
                except (ValueError, TypeError):
                    latest_order_date = None
                if latest_order_date is not None:
                    as_of_date = st.date_input(
                        "Measure activity as of",
                        value=latest_order_date,
                        min_value=latest_order_date,
                        help="Defaults to the latest order date in your file, "
                        "so historical exports are not mistaken for current activity.",
                    )
                else:
                    as_of_date = None
                    st.warning("The date column could not be interpreted. Check the column mapping.")
                submitted = st.form_submit_button(
                    "Analyse order history", type="primary",
                    disabled=as_of_date is None,
                )
            if submitted:
                report_df, report_meta = summarize_orders(
                    orders_df,
                    customer_col=customer_col,
                    date_col=date_col,
                    order_col=None if order_col_choice == order_options[0] else order_col_choice,
                    amount_col=None if amount_col_choice == amount_options[0] else amount_col_choice,
                    amount_type="order_total" if amount_kind.startswith("Order total") else "line_total",
                    reference_date=as_of_date,
                    inactive_days=inactivity_days,
                )
                st.session_state.store_orders_summary = report_df
                st.session_state.store_orders_metadata = report_meta
                st.session_state.store_orders_filename = orders_upload.name
                try:
                    historical_events = canonical_orders(
                        orders_df,
                        customer_col=customer_col,
                        order_col=None if order_col_choice == order_options[0] else order_col_choice,
                        date_col=date_col,
                    )
                    readiness, historical_examples = assess_prediction_readiness(historical_events)
                    st.session_state.store_orders_readiness = readiness
                    st.session_state.store_orders_examples = historical_examples
                except (ValueError, KeyError, TypeError) as error:
                    st.session_state.store_orders_readiness = {"error": str(error)}
                    st.session_state.store_orders_examples = None
                st.success("Order history analysed. These are observed facts, not AI predictions.")
        except Exception as error:
            st.error(f"Unable to read or analyse this orders file: {error}")

    store_summary = st.session_state.get("store_orders_summary")
    store_meta = st.session_state.get("store_orders_metadata")
    if store_summary is not None and store_meta is not None:
        st.divider()
        st.subheader("Your store's buying activity")
        st.caption(
            f"Source: {st.session_state.get('store_orders_filename', 'Uploaded file')} · "
            f"As of {store_meta['reference_date']} · "
            f"Recent activity means an order within {store_meta['inactive_days']} days"
        )
        first, second, third = st.columns(3)
        first.metric("Customers in file", f"{store_meta['customers']:,}")
        second.metric("Orders counted", f"{store_meta['orders']:,}")
        third.metric(
            f"No order in {store_meta['inactive_days']}+ days",
            f"{store_meta['customers_without_recent_order']:,}",
        )
        if store_meta["has_amount"]:
            st.metric(
                "Order value in file (same currency as upload)",
                f"{store_summary['Total order value'].sum():,.2f}",
            )
        if store_meta["no_order_id"]:
            st.warning(
                "No order ID was selected, so CHURNIQ counted each transaction row as a separate order. "
                "This can overcount orders when an export has multiple rows per order."
            )
        st.caption(
            "The values include all uploaded rows. Remove cancelled, refunded and test orders "
            "before importing if they should not count as purchases. "
            "Use a single currency when interpreting order totals."
        )
        plot = px.histogram(
            store_summary, x="Days since last order", nbins=20,
            title="How long has it been since customers last ordered?",
            labels={"Days since last order": "Days since latest recorded order"},
            template="plotly_dark",
        )
        plot.update_layout(yaxis_title="Number of customers")
        st.plotly_chart(plot, use_container_width=True)
        st.subheader("Customers who have not ordered recently")
        review = store_summary[
            store_summary["Days since last order"] >= store_meta["inactive_days"]
        ]
        st.dataframe(review.head(100), hide_index=True, use_container_width=True)
        st.download_button(
            "Download order-history summary",
            store_summary.to_csv(index=False),
            file_name="churniq_store_order_summary.csv", mime="text/csv",
        )
        st.caption(
            "This identifies past ordering behaviour only. It cannot tell you "
            "whether a customer will make a purchase in the next four weeks."
        )
        st.divider()
        st.subheader("Is this order history ready for AI predictions?")
        st.write(
            "A real four-week prediction model needs enough **older purchases** to learn from "
            "and **later purchases** to check its answers. We check whether your file could "
            "support that type of experiment; this page does not train or apply a model."
        )
        readiness = st.session_state.get("store_orders_readiness")
        if readiness is None:
            st.info("Run the order-history analysis above to inspect training readiness.")
        elif "error" in readiness:
            st.warning(f"Training-readiness check could not be completed: {readiness['error']}")
        else:
            left, middle, right = st.columns(3)
            left.metric("Order history", f"{readiness['history_days']} days")
            middle.metric("Distinct buyers", f"{readiness['unique_customers']:,}")
            right.metric("Historical training records", f"{readiness['eligible_customer_periods']:,}")
            if readiness["passes_basic_checks"]:
                st.success(
                    "This file passes the initial size and history checks. "
                    "A separately validated model-training procedure is still required."
                )
            else:
                st.warning(
                    "Not enough evidence to train a reliable four-week prediction model from this file yet. "
                    "Use a longer paid-order export with more customers and repeat purchases."
                )
            for criterion, passed in readiness["minimum_checks"].items():
                marker = "✓" if passed else "✗"
                st.write(f"{marker} {criterion}")
            with st.expander("How future-purchase examples would be prepared"):
                st.write(
                    "At each historical checkpoint, CHURNIQ would calculate order counts and "
                    "how recently each customer bought using **only past orders**. "
                    "It would then inspect the next 28 days to label whether a purchase occurred. "
                    "The resulting labels are for retrospective training, not predictions."
                )
                st.write(
                    f"Examples with no order in the following four weeks: "
                    f"{readiness['examples_no_future_purchase']:,}. "
                    f"Examples with an order: {readiness['examples_with_future_purchase']:,}."
                )
                examples = st.session_state.get("store_orders_examples")
                if examples is not None and not examples.empty:
                    st.dataframe(examples.head(10), hide_index=True, use_container_width=True)
            st.caption(
                "These checks are illustrative, not a validation certificate. "
                "The store must confirm that the export covers every paid order for the "
                "complete 28-day future window; the last order date alone cannot prove that. "
                "Cancelled and refunded orders must be filtered correctly. "
                "No REES46 XGBoost model or calibrated probability is applied to these rows."
            )

elif page == "Check Customers":
    title_block(
        "START HERE",
        "Find customers who may not buy again",
        "Upload a prepared customer-activity file and get an easy-to-read priority list. "
        "No AI knowledge is needed to interpret the results.",
    )
    if bundle is None:
        st.error("The prediction model has not been loaded. Check that artifacts/churniq_rees46.joblib exists.")
    else:
        st.info(
            "What you need: **a prepared customer-activity CSV**, not just a list of orders. "
            "This research model uses historical shopping behaviour that must be calculated "
            "from your store's events and transactions first."
        )
        upload_tab, sample_tab = st.tabs(["Upload your prepared file", "Try an example"])
        with upload_tab:
            uploaded = st.file_uploader("Choose customer-activity CSV", type="csv")
            if uploaded is not None:
                try:
                    data = load_csv(uploaded)
                    st.write(f"File ready: **{len(data):,} customer records**.")
                    st.dataframe(data.head(5).iloc[:, :8], hide_index=True, use_container_width=True)
                    missing = [field for field in bundle["features"] if field not in data.columns]
                    if missing:
                        st.warning(f"{len(missing)} required activity fields are missing.")
                        with st.expander("See the technical fields required"):
                            st.write(missing)
                    else:
                        possible_ids = [col for col in ("user_id", "customer_id", "customerid", "visitorid") if col in data.columns]
                        id_options = possible_ids + ["Use automatic record numbers"]
                        chosen = st.selectbox("Which column identifies your customers?", id_options)
                        if "target_event" in data.columns:
                            st.warning(
                                "This file contains historical outcome labels. They are not used by "
                                "the model, but this is a retrospective check, not a future prediction demonstration."
                            )
                        if st.button("Check customer risk", type="primary", key="upload_predict"):
                            if run_customer_predictions(
                                data, uploaded.name, None if chosen == id_options[-1] else chosen
                            ):
                                st.success("Customer analysis complete. View results below or on Business Home.")
                except Exception as error:
                    st.error(f"Could not read the customer file: {error}")
        with sample_tab:
            st.write(
                "Explore CHURNIQ with prepared **historical held-out REES46 records**. "
                "They represent past records excluded from the original grouped model's training "
                "and are not live customers."
            )
            if SAMPLE_PATH.exists():
                if st.button("Analyse example customers", key="sample_predict"):
                    try:
                        sample = load_csv(SAMPLE_PATH)
                        if run_customer_predictions(sample, "REES46 historical example", "user_id"):
                            st.success("Example records analysed. See your results below.")
                    except Exception as error:
                        st.error(f"Could not read the example: {error}")
            else:
                st.caption(
                    "The local example file is not available. Generate it using "
                    "python -m scripts.export_holdout_cohort, or upload a prepared file."
                )
        result, _ = session_predictions()
        if result is not None:
            st.divider()
            st.subheader("Customer results")
            st.write(
                "Start with **Needs attention**, review the reasons in "
                "**Why Customers Are Flagged**, then use **Follow-up Planner**."
            )
            show_customer_table(result, limit=20)
            risk_note()


elif page == "Customer List":
    title_block(
        "CUSTOMER PRIORITIES",
        "Your customer review list",
        "Search customers, focus on an attention level, and export a list for your team.",
    )
    result, _ = predictions_or_help()
    if result is not None:
        display = display_predictions(result)
        chosen_levels = st.multiselect(
            "Show customers marked",
            ["Needs attention", "Keep an eye on", "Lower priority"],
            default=["Needs attention", "Keep an eye on", "Lower priority"],
        )
        keyword = st.text_input("Search customer ID", placeholder="Enter any part of the customer ID")
        chosen = display[display["Priority"].isin(chosen_levels)]
        if keyword:
            chosen = chosen[chosen["Customer ID"].str.contains(keyword, case=False, regex=False)]
        chosen = chosen.sort_values("Risk score (not calibrated)", ascending=False)
        st.caption(f"{len(chosen):,} matching customer records")
        st.dataframe(chosen, use_container_width=True, hide_index=True)
        if not chosen.empty:
            st.subheader("Understand one customer")
            chosen_idx = st.selectbox(
                "Choose a record to explain",
                options=chosen.index.tolist(),
                format_func=lambda idx: (
                    f"{result.iloc[idx]['customer_id']} — "
                    f"{display_priority(result.iloc[idx]['risk_band'])}, "
                    f"score {result.iloc[idx]['predicted_no_purchase_risk_pct']:.1f}/100"
                ),
                key="list_explain_customer",
            )
            st.button(
                "Show why this customer was flagged →",
                on_click=go_to,
                args=("Why Customers Are Flagged", int(chosen_idx)),
                type="primary",
                key="list_explain_button",
            )
        st.download_button(
            "Download this customer list (CSV)",
            chosen.to_csv(index=False),
            file_name="churniq_customer_review.csv",
            mime="text/csv",
        )
        risk_note()


elif page == "Why Customers Are Flagged":
    title_block(
        "UNDERSTAND THE SIGNALS",
        "What influenced this customer's risk?",
        "Understand which shopping and purchasing patterns influenced the model's score. "
        "These are clues to investigate, not proven reasons someone stopped buying.",
    )
    result, raw = predictions_or_help()
    if bundle is not None and result is not None:
        selected = st.selectbox(
            "Choose a customer record",
            range(len(result)),
            index=min(int(st.session_state.get("focused_customer_idx", 0)), len(result) - 1),
            format_func=lambda idx: (
                f"{result.iloc[idx]['customer_id']} · "
                f"{result.iloc[idx]['predicted_no_purchase_risk_pct']:.1f}/100 risk score"
            ),
        )
        st.session_state["focused_customer_idx"] = int(selected)
        selected_row = result.iloc[selected]
        col1, col2 = st.columns(2)
        col1.metric("Model risk score", f"{selected_row['predicted_no_purchase_risk_pct']:.1f} / 100")
        col2.metric("Review priority", display_priority(selected_row["risk_band"]))
        risk_note()
        try:
            detail = explain(bundle, raw.iloc[[selected]], max_rows=1)
            local = top_local(detail, 0).copy()
            local["Shopping signal"] = local["feature"].map(readable_feature)
            local["Effect on score"] = local["contribution_log_odds"].map(readable_effect)
            local["Strength"] = local["contribution_log_odds"].abs()
            most = local.sort_values("Strength", ascending=False).head(5)
            increased = local[local["contribution_log_odds"] > 0].sort_values(
                "contribution_log_odds", ascending=False
            )
            important_signals = list(dict.fromkeys(increased["Shopping signal"].tolist()))[:2]
            st.subheader("What this means for your store")
            if important_signals:
                st.write(
                    "The model flagged shopping patterns related to "
                    + " and ".join(f"**{name.lower()}**" for name in important_signals)
                    + " as important signals behind this risk estimate."
                )
            else:
                st.write(
                    "No single risk-increasing shopping pattern stands out in the "
                    "top explanation. Review the customer's history before acting."
                )
            st.markdown("#### What should your team check?")
            review_prompts = suggested_review_checks(
                list(zip(local["feature"], local["contribution_log_odds"]))
            )
            for prompt in review_prompts:
                st.markdown("- " + prompt)
            st.caption(
                "These are suggested checks based on model signals. They do not "
                "prove why a customer might stop buying or guarantee a successful follow-up."
            )
            st.subheader("Shopping patterns that influenced the estimate")
            st.dataframe(
                most[["Shopping signal", "Effect on score"]],
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "The model may raise or lower a customer's risk score based on each "
                "pattern. These effects are model explanations, not causes."
            )
            with st.expander("Explore the detailed influence chart (advanced)"):
                diagram = local.sort_values("contribution_log_odds").copy()

                diagram["Impact"] = diagram["contribution_log_odds"].map(
                    lambda value: "Raised estimated risk" if value > 0 else "Lowered estimated risk"
                )
                figure = px.bar(
                    diagram, x="contribution_log_odds", y="Shopping signal", orientation="h",
                    color="Impact",
                    color_discrete_map={
                        "Raised estimated risk": "#f68da0",
                        "Lowered estimated risk": "#40d6ba",
                    },
                    hover_data={"feature": True, "contribution_log_odds": ":.3f"},
                    template="plotly_dark",
                    labels={
                        "contribution_log_odds": "Influence on model score (not probability points)"
                    },
                )
                figure.update_layout(legend_title_text=None, height=500)
                st.plotly_chart(figure, use_container_width=True)
            st.info(
                "Before taking action, review the customer's actual order history, "
                "shopping activity and service interactions. A model explanation "
                "is not evidence that a specific event caused the prediction."
            )
            st.button(
                "Open follow-up planner →",
                on_click=go_to, args=("Follow-up Planner",),
                key="explanation_to_planner",
                type="primary",
            )
            with st.expander("Which signals matter across all uploaded records?"):
                global_detail = explain(bundle, raw, max_rows=min(150, len(raw)))
                important = top_global(global_detail).copy()
                important["Shopping signal"] = important["feature"].map(readable_feature)
                st.plotly_chart(
                    px.bar(
                        important.sort_values("mean_abs_shap_log_odds"),
                        x="mean_abs_shap_log_odds",
                        y="Shopping signal", orientation="h", template="plotly_dark",
                        labels={
                            "mean_abs_shap_log_odds": "Average influence (model units)"
                        },
                        hover_data={"feature": True},
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "This chart summarises the sample of uploaded records only. "
                    "It measures influence, not whether a signal increases or decreases risk."
                )
            with st.expander("Technical explanation (for researchers)"):
                st.write("Method: SHAP. The underlying contributions are in model log-odds, not percentage points.")
                st.dataframe(
                    local[["feature", "contribution_log_odds", "model_input_value"]],
                    use_container_width=True, hide_index=True,
                )
        except Exception as error:
            st.error(f"Could not explain this prediction: {error}")


elif page == "Follow-up Planner":
    title_block(
        "TURN INSIGHTS INTO A PLAN",
        "Decide who needs a closer look",
        "Use these suggestions as a starting point for your retention team, "
        "not as automatic marketing instructions.",
    )
    result, _ = predictions_or_help()
    if result is not None:
        high = result[result["risk_band"] == "High"]
        watch = result[result["risk_band"] == "Moderate"]
        col1, col2 = st.columns(2)
        col1.metric("Review first", len(high))
        col2.metric("Keep an eye on", len(watch))
        st.subheader("Suggested next steps")
        with st.container(border=True):
            st.markdown("#### 🔴 Needs attention")
            st.write(suggested_review_action("High"))
            st.caption("These records have a model score of 70 or higher out of 100.")
        with st.container(border=True):
            st.markdown("#### 🟡 Keep an eye on")
            st.write(suggested_review_action("Moderate"))
            st.caption("These records have a model score from 40 up to 70.")
        with st.container(border=True):
            st.markdown("#### 🟢 Lower priority")
            st.write(suggested_review_action("Low"))
        st.markdown(
            "**Before customer outreach:** check whether it is appropriate, "
            "respect marketing consent and privacy requirements, and consider "
            "whether the customer recently purchased or contacted support."
        )
        st.subheader("Download a manual review shortlist")
        priority_table = display_predictions(result[result["risk_band"].isin(["High", "Moderate"])])
        priority_table = priority_table.sort_values("Risk score (not calibrated)", ascending=False)
        st.dataframe(priority_table.head(30), use_container_width=True, hide_index=True)
        st.download_button(
            "Download follow-up review list",
            priority_table.to_csv(index=False),
            file_name="churniq_followup_review.csv", mime="text/csv",
        )
        st.caption("CHURNIQ does not contact customers or send promotions automatically.")
        risk_note()


elif page == "Research Lab":
    title_block(
        "FOR RESEARCHERS AND SUPERVISORS",
        "Model evidence and technical details",
        "The business pages keep things simple. This area preserves the full "
        "model comparison, experiment results and technical settings for your thesis.",
    )
    evaluation_tab, calibration_tab, temporal_tab, details_tab = st.tabs(
        ["Model comparison", "Calibration study", "Future-period test", "Technical details"]
    )
    with evaluation_tab:
        if bundle is None:
            st.info("No saved model is currently loaded.")
        else:
            report = bundle["report"]
            st.info(
                "These scores belong to the model artifact currently loaded in the dashboard. "
                "They do not automatically describe the separately trained full candidate."
            )
            scores = pd.DataFrame(report["scores"]).T
            if "confusion_matrix" in scores:
                scores = scores.drop(columns=["confusion_matrix"])
            st.dataframe(scores, use_container_width=True)
            curve = px.line(template="plotly_dark")
            for name, values in report.get("roc_curves", {}).items():
                curve.add_scatter(x=values["fpr"], y=values["tpr"], mode="lines", name=name)
            curve.update_layout(
                title="ROC curves for the loaded model's training experiment",
                xaxis_title="False positive rate", yaxis_title="True positive rate",
            )
            st.plotly_chart(curve, use_container_width=True)
            st.caption("Customer-group held-out evaluation is not the same as out-of-time evaluation.")

    with calibration_tab:
        st.warning(
            "These results come from a separate experimental model. "
            "They do NOT calibrate the predictions shown on the business pages, "
            "and they are not a future-period calibration test."
        )
        if CALIBRATION_PATH.exists():
            try:
                study = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
                if (study.get("experiment") != "exploratory_group_disjoint_sigmoid_calibration"
                        or study.get("fast") is not False):
                    st.error("The saved file is not the full customer-disjoint calibration report.")
                else:
                    raw_metrics = study["raw_test"]
                    calibrated = study["calibrated_test"]
                    a, b, c = st.columns(3)
                    a.metric("Original ECE", f"{raw_metrics['ece_10_bins']:.2%}")
                    b.metric("Calibrated ECE", f"{calibrated['ece_10_bins']:.2%}")
                    c.metric("Calibrated Brier score", f"{calibrated['brier']:.4f}")
                    st.caption(
                        f"{study['train_customers']:,} training customers · "
                        f"{study['calibration_customers']:,} calibration customers · "
                        f"{study['test_customers']:,} evaluation customers (all distinct)"
                    )
                    chosen_metrics = [
                        ("Accuracy", "accuracy"), ("Precision", "precision"),
                        ("Recall", "recall"), ("F1-score", "f1"),
                        ("ROC-AUC", "roc_auc"), ("Brier score", "brier"),
                        ("ECE", "ece_10_bins"),
                    ]
                    comparison = pd.DataFrame([
                        {"Metric": label, "Before": raw_metrics[key], "After": calibrated[key]}
                        for label, key in chosen_metrics
                    ])
                    st.dataframe(
                        comparison.style.format({"Before": "{:.4f}", "After": "{:.4f}"}),
                        use_container_width=True, hide_index=True,
                    )
                    plot = pd.DataFrame({
                        "Stage": ["Uncalibrated", "Sigmoid-calibrated"],
                        "Calibration error (%)": [
                            raw_metrics["ece_10_bins"] * 100,
                            calibrated["ece_10_bins"] * 100,
                        ],
                    })
                    st.plotly_chart(
                        px.bar(
                            plot, x="Stage", y="Calibration error (%)",
                            title="Expected Calibration Error (lower is better)",
                            template="plotly_dark",
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "This is an exploratory pooled-period experiment. "
                        "It cannot establish calibration reliability on future periods."
                    )
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f"Calibration report could not be shown: {error}")
        else:
            st.info(
                "No full calibration report found. Place "
                "rees46_calibration_full.json in data/reports/ to display the experiment."
            )

    with temporal_tab:
        st.warning(
            "This report is from a separately trained historical-only experiment, "
            "not the model currently serving predictions."
        )
        if TEMPORAL_PATH.exists():
            try:
                historical = json.loads(TEMPORAL_PATH.read_text(encoding="utf-8"))
                if historical.get("experiment") != "out_of_time_rees46_period_0" or historical.get("fast") is not False:
                    st.error("Expected the full out-of-time experiment report.")
                else:
                    metrics = historical["temporal_test"]
                    cols = st.columns(4)
                    for col, (label, key) in zip(
                        cols, [
                            ("F1-score", "f1"), ("Recall", "recall"),
                            ("ROC-AUC", "roc_auc"), ("Brier score", "brier"),
                        ],
                    ):
                        col.metric(label, f"{metrics[key]:.3f}")
                    st.write(
                        f"Training pool: {historical['older_rows']:,} records from earlier periods. "
                        f"Newer evaluation period: {historical['test_rows']:,} records."
                    )
                    st.write(
                        "The newest period contains a different mix of purchasing "
                        "and non-purchasing customers. This affects performance and calibration."
                    )
                    with st.expander("Full time-based evaluation details"):
                        st.json(historical)
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f"Temporal report could not be shown: {error}")
        else:
            st.info("Full temporal report not found at data/reports/rees46_temporal_full.json.")

    with details_tab:
        if bundle is not None:
            report = bundle["report"]
            st.write("**Loaded model:**", report.get("model_used", "Unspecified"))
            st.write("**Target:** No purchase during the following four weeks (label 1).")
            st.write("**Training records:**", f"{report.get('source_rows', 0):,}")
            st.write("**Data split:**", report.get("split", "Not reported"))
            st.json({
                "selected_model_parameters": report.get("best_params", {}),
                "selected_input_features": report.get("selected_transformed_features", []),
                "target_column": report.get("target_column"),
            })
            st.caption(
                "SHAP shows model attribution, not the cause of customer behaviour. "
                "The current model's risk scores are not production-calibrated."
            )
        st.info(
            "For reproducibility, train models using the offline CLI scripts. "
            "This dashboard intentionally avoids overwriting the active model from a browser."
        )


elif page == "Settings":
    title_block(
        "PREFERENCES",
        "Review settings",
        "Control how CHURNIQ labels its predictions, or clear the current customer analysis.",
    )
    st.subheader("How customer priorities are assigned")
    st.write(
        "**Needs attention:** score of 70–100.  "
        "**Keep an eye on:** score of 40–69.9.  "
        "**Lower priority:** below 40."
    )
    st.caption("These are demonstration grouping rules, not calibrated probabilities or proven campaign thresholds.")
    with st.expander("Advanced: model classification threshold"):
        value = st.slider(
            "Threshold for the model's yes/no four-week prediction",
            0.05, 0.95, float(st.session_state.get("decision_threshold", 0.5)), 0.05,
        )
        st.session_state.decision_threshold = value
        st.caption(
            "This only changes the model's yes/no classification when you NEXT run "
            "customer predictions. It does not change the three priority groups or calibrate risk scores."
        )
    if bundle is not None:
        with st.expander("Technical input requirements"):
            st.write(f"Prepared activity features required: {len(bundle['features'])}")
            st.dataframe(pd.DataFrame({"Feature name": bundle["features"]}), hide_index=True)
    if st.button("Clear analysed customers"):
        st.session_state.pop("predictions", None)
        st.session_state.pop("customers", None)
        st.session_state.pop("prediction_source", None)
        st.success("Customer results cleared from this session.")


st.divider()
st.caption(
    "CHURNIQ · UiTM research prototype · No customer outreach is automated · "
    "Scores are uncalibrated model estimates · Results are not proof of causation"
)
