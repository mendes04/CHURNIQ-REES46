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
from churniq.ui import CSS, hero, section, cards, guide
from churniq.thesis_alignment import (
    PROJECT_TITLE, CHURN_PROXY_NOTE, TARGET_RECONCILIATION_NOTE,
    THESIS_CHURN_DEFINITION, MODEL_CHURN_DEFINITION, model_artifact_label,
)


ROOT = Path(__file__).parent
FULL_MODEL_PATH = ROOT / "artifacts" / "churniq_full_candidate.joblib"
DEMO_MODEL_PATH = ROOT / "artifacts" / "churniq_rees46.joblib"
MODEL_PATH = FULL_MODEL_PATH if FULL_MODEL_PATH.exists() else DEMO_MODEL_PATH
SAMPLE_PATH = (
    ROOT / "data" / "sample" / "rees46_chapter4_holdout.csv"
    if MODEL_PATH == FULL_MODEL_PATH else
    ROOT / "data" / "sample" / "rees46_heldout_cohort.csv"
)
CALIBRATION_PATH = ROOT / "data" / "reports" / "rees46_calibration_full.json"
TEMPORAL_PATH = ROOT / "data" / "reports" / "rees46_temporal_full.json"

st.set_page_config(page_title="CHURNIQ | Explainable Customer Churn Prediction", page_icon="🛍️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


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
        CHURN_PROXY_NOTE
    )


def title_block(eyebrow, title, description):
    st.markdown(hero(eyebrow, title, description), unsafe_allow_html=True)


def section_block(title, description=""):
    st.markdown(section(title, description), unsafe_allow_html=True)


def card_grid(items, columns=3, metric=False):
    st.markdown(cards(items, columns=columns, metric=metric), unsafe_allow_html=True)


def helpful_note(title, description):
    st.markdown(guide(title, description), unsafe_allow_html=True)


def show_customer_table(result, limit=None):
    view = display_predictions(result)
    view = view.sort_values("Churn risk score (not calibrated)", ascending=False)
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
st.sidebar.markdown(
    '<div class="cq-brand"><div class="cq-brand-title">✧ CHURN<span>IQ</span></div>'
    '<div class="cq-brand-caption">EXPLAINABLE CUSTOMER CHURN PREDICTION</div></div>',
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Explore CHURNIQ",
    [
        "Business Home",
        "Check Customers",
        "Customer List",
        "Why Customers Are Flagged",
        "Follow-up Planner",
        "Store Orders (Preview)",
        "Research Lab",
        "Settings",
    ],
    key="navigation_page",
    format_func=lambda name: {
        "Business Home": "⌂  Churn Overview",
        "Check Customers": "◎  Predict Churn",
        "Customer List": "☷  Churn Risk List",
        "Why Customers Are Flagged": "◇  Explain Churn (SHAP)",
        "Follow-up Planner": "✦  Retention Planner",
        "Store Orders (Preview)": "▣  Store Orders",
        "Research Lab": "◈  Model Evaluation",
        "Settings": "⚙  Settings",
    }.get(name, name),
)
if bundle is not None:
    if MODEL_PATH == FULL_MODEL_PATH:
        st.sidebar.success("Full tuned XGBoost model loaded")
    else:
        st.sidebar.warning("Earlier demo model loaded; full candidate not found")
else:
    st.sidebar.warning("REES46 prediction model not loaded")
st.sidebar.markdown(
    '<div class="cq-side-note"><strong>New here?</strong>'
    '<p>1. Check customers using a prepared activity file.<br>'
    '2. Read the results and review customers.<br>'
    '3. Use Store Orders for normal shop order files.</p></div>',
    unsafe_allow_html=True,
)
with st.sidebar.expander("What do the pages do?"):
    st.markdown(
        "**Home:** summary and next steps.  \n"
        "**Check Customers:** run the trained research model.  \n"
        "**Customer List:** search and download results.  \n"
        "**Why Flagged:** understand the model's signals.  \n"
        "**Follow-up:** decide what to review next.  \n"
        "**Store Orders:** explore ordinary order exports without AI predictions.  \n"
        "**Research Lab:** technical results for the thesis.  \n"
        "**Settings:** preferences and session data."
    )
st.sidebar.caption("Research prototype · No automatic customer messages")
st.sidebar.caption("Model source: " + MODEL_PATH.name)


if page == "Business Home":
    title_block(
        "YOUR RETENTION OVERVIEW",
        "Understand and predict customer churn",
        "Identify customers at risk of churn, understand their shopping patterns "
        "with SHAP, and support better retention decisions."
    )
    result, _ = session_predictions()
    if result is None:
        helpful_note("About churn in this research:", CHURN_PROXY_NOTE)
        section_block(
            "Start with three simple steps",
            "You do not need to understand AI to use the business pages.",
        )
        card_grid([
            {"icon": "📂", "label": "1. Choose your data",
             "description": "Use a prepared customer activity file for AI risk checks. "
                            "Have a normal orders CSV? Open Store Orders instead.", "tone": "purple"},
            {"icon": "🔎", "label": "2. Check the results",
             "description": "See which records are in the highest review-priority group "
                            "and which may need monitoring.", "tone": "teal"},
            {"icon": "💜", "label": "3. Decide what to do",
             "description": "Read the model's signals, check actual customer history "
                            "and download a list for your team.", "tone": "amber"},
        ])
        left, right = st.columns(2, gap="medium")
        with left:
            st.button(
                "Predict customer churn →",
                type="primary", use_container_width=True,
                on_click=go_to, args=("Check Customers",), key="home_start_predictions",
            )
        with right:
            st.button(
                "Explore store orders →",
                use_container_width=True,
                on_click=go_to, args=("Store Orders (Preview)",), key="home_start_orders",
            )
        if bundle is None:
            st.warning(
                "The REES46 prediction model is not loaded yet. "
                "You can still explore normal order-history data."
            )
        else:
            st.info(
                "For a guided example, open Check Customers → Try an example. "
                "It uses historical REES46 records if your local sample file is available."
            )
    else:
        total = len(result)
        unique_customers = result["customer_id"].nunique()
        high = int((result["risk_band"] == "High").sum())
        watch = int((result["risk_band"] == "Moderate").sum())
        low = int((result["risk_band"] == "Low").sum())
        average_score = float(result["predicted_no_purchase_risk"].mean()) * 100
        st.caption(
            f"Showing: {st.session_state.get('prediction_source', 'Current analysis')} "
            f"· {total:,} records · {unique_customers:,} unique customer IDs"
        )
        section_block(
            "Churn risk overview",
            "These numbers describe only the records analysed in this session, "
            "not every customer in your online store.",
        )
        card_grid([
            {"icon": "👥", "label": "Records checked", "value": f"{total:,}",
             "description": "Customer records in this analysis.", "tone": "purple"},
            {"icon": "🚩", "label": "High churn risk", "value": f"{high:,}",
             "description": "Model score of 70 or more out of 100.", "tone": "rose"},
            {"icon": "👀", "label": "Moderate churn risk", "value": f"{watch:,}",
             "description": "Model score from 40 to under 70.", "tone": "amber"},
            {"icon": "📊", "label": "Average churn risk score", "value": f"{average_score:.1f} / 100",
             "description": "Not an exact real-world chance of leaving.", "tone": "teal"},
        ], columns=4, metric=True)
        if high:
            helpful_note(
                "Where to begin:",
                f"{high:,} records are in the highest review group. "
                "Check real purchasing and support history before contacting anyone."
            )
        else:
            helpful_note(
                "Good to know:",
                "No records are in the highest review group in this file. "
                "Keep monitoring activity and continue normal customer care."
            )
        section_block(
            "Which customers have the highest churn risk?",
            "Review-priority groups are simple demonstration ranges based on model scores.",
        )
        left, right = st.columns([1, 1.35], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("#### Review priority breakdown")
                st.caption("How many analysed records fall into each priority group?")
                counts = pd.DataFrame({
                    "Priority": ["Needs attention", "Keep an eye on", "Lower priority"],
                    "Customer records": [high, watch, low],
                })
                fig = px.bar(
                    counts, x="Priority", y="Customer records", color="Priority",
                    text="Customer records", template="plotly_dark",
                    color_discrete_map={
                        "Needs attention": "#f58da4",
                        "Keep an eye on": "#efbf76",
                        "Lower priority": "#5ad7c4",
                    }
                )
                fig.update_layout(
                    height=318, showlegend=False, xaxis_title=None,
                    margin=dict(t=12, l=8, r=8, b=4),
                )
                st.plotly_chart(fig, use_container_width=True)
        with right:
            with st.container(border=True):
                st.markdown("#### Customers to review first")
                st.caption("Highest model scores appear first. Open the list to investigate.")
                show_customer_table(result, limit=8)
                st.button(
                    "Open the full customer list →", type="primary",
                    on_click=go_to, args=("Customer List",),
                    key="home_customer_list", use_container_width=True,
                )
        section_block(
            "Understand the scores",
            "A score helps you decide whom to review. "
            "It does not prove that someone will stop buying.",
        )
        card_grid([
            {"icon": "🔴", "label": "Needs attention — 70 to 100",
             "description": "Start by reviewing these customers' recent orders, "
                            "visits and any support requests.", "tone": "rose"},
            {"icon": "🟠", "label": "Keep an eye on — 40 to under 70",
             "description": "Monitor changes in shopping activity and check "
                            "whether the customer needs help.", "tone": "amber"},
            {"icon": "🟢", "label": "Lower priority — below 40",
             "description": "Continue normal service. A lower score is not "
                            "a promise of another purchase.", "tone": "teal"},
        ])
        section_block(
            "See the details behind the numbers",
            "The chart uses the current model scores, not invented weekly trends.",
        )
        lower_left, lower_right = st.columns([1.15, 1], gap="large")
        with lower_left:
            with st.container(border=True):
                st.markdown("#### Churn risk score distribution")
                st.caption("How many records have low, medium or high model scores?")
                score_values = pd.DataFrame({
                    "Churn risk score": result["predicted_no_purchase_risk"].astype(float) * 100
                })
                fig_dist = px.histogram(
                    score_values, x="Churn risk score", nbins=20, template="plotly_dark",
                    color_discrete_sequence=["#a184f9"],
                )
                fig_dist.update_layout(
                    height=300, xaxis_title="Score out of 100",
                    yaxis_title="Records", showlegend=False,
                    margin=dict(t=20, l=8, r=8, b=8),
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        with lower_right:
            with st.container(border=True):
                st.markdown("#### Your next actions")
                st.write("**1. Check a customer.** See the shopping patterns the model used.")
                st.write("**2. Review real history.** Check recent purchases, visits and service issues.")
                st.write("**3. Plan a response.** Decide if a follow-up is suitable.")
                st.button(
                    "Understand a customer →", on_click=go_to,
                    args=("Why Customers Are Flagged",),
                    key="home_explain_button", use_container_width=True,
                )
                st.button(
                    "Open follow-up planner →", on_click=go_to,
                    args=("Follow-up Planner",),
                    key="home_plan_button", use_container_width=True,
                )
        risk_note()

    section_block(
        "Your store orders, separately",
        "Normal orders CSV files show what customers actually bought. "
        "This is not the same as running the REES46 prediction model.",
    )
    order_summary = st.session_state.get("store_orders_metadata")
    if order_summary is None:
        card_grid([
            {"icon": "🛍️", "label": "Order history",
             "description": "Upload your store's paid orders and inspect recent "
                            "purchasing patterns.", "tone": "purple"},
            {"icon": "🗓️", "label": "Customers without a recent order",
             "description": "Choose a number of days to find customers who "
                            "have not purchased recently.", "tone": "amber"},
            {"icon": "📑", "label": "Export a customer summary",
             "description": "Download observed order counts and dates. "
                            "This is not an AI prediction.", "tone": "teal"},
        ])
    else:
        card_grid([
            {"icon": "🛍️", "label": "Orders counted",
             "value": f"{order_summary['orders']:,}",
             "description": "Orders in your uploaded transaction file.", "tone": "purple"},
            {"icon": "👤", "label": "Customers in orders file",
             "value": f"{order_summary['customers']:,}",
             "description": "Different customer IDs in that file.", "tone": "teal"},
            {"icon": "🕓", "label": f"No orders in {order_summary['inactive_days']}+ days",
             "value": f"{order_summary['customers_without_recent_order']:,}",
             "description": f"Observed as of {order_summary['reference_date']}.", "tone": "amber"},
        ], metric=True)
        st.caption(
            "Order-history counts use a different uploaded file from the customer-risk model. "
            "They are not connected customer-level predictions."
        )
    st.button(
        "Go to Store Orders →", on_click=go_to, args=("Store Orders (Preview)",),
        key="home_order_button",
    )
    section_block(
        "Explore what each page can do",
        "Each section answers one practical question for your online business.",
    )
    card_grid([
        {"icon": "🔎", "label": "Predict Churn",
         "description":  "Which prepared customer records show "
                        "higher churn risk based on historical shopping behaviour?", "tone": "purple"},
        {"icon": "📋", "label": "Customer List",
         "description": "Who should I look at first, and how do I export "
                        "a review list?", "tone": "amber"},
        {"icon": "🧩", "label": "Explain Churn (SHAP)",
         "description": "Which shopping patterns affected the model score "
                        "for one customer?", "tone": "teal"},
        {"icon": "💌", "label": "Follow-up Planner",
         "description": "What might my team review before choosing "
                        "a customer-friendly action?", "tone": "rose"},
        {"icon": "🛒", "label": "Store Orders",
         "description": "What has happened in actual order records, "
                        "without making AI predictions?", "tone": "purple"},
        {"icon": "🧪", "label": "Research Lab",
         "description": "How was the model tested? See technical metrics "
                        "and separate experiments for research.", "tone": "teal"},
    ], columns=3)
    with st.expander("Detailed guide: How to use every CHURNIQ page"):
        st.markdown(
            "**Business Home:** Start here after checking customer records. "
            "The boxes show how many records need attention, while the charts help you "
            "spot which scores are most common in your uploaded file.\n\n"
            "**Check Customers:** Upload a prepared REES46-style customer activity CSV. "
             "The model uses earlier shopping patterns to estimate a churn-class score. "
             "The current code describes this outcome as no purchase in four weeks. An ordinary order export "
            "cannot be used here yet.\n\n"
            "**Customer List:** Search IDs, focus on a review group and export a CSV. "
            "Start with higher scores, but verify the customer's actual history first.\n\n"
            "**Why Customers Are Flagged:** Choose one customer to see which earlier "
            "shopping signals raised or lowered the model score. These are model "
            "explanations, not proven reasons for the customer's behaviour.\n\n"
            "**Follow-up Planner:** Consider customer-friendly review actions such as "
            "checking orders, browsing patterns or support issues. The system never "
            "sends messages automatically.\n\n"
            "**Store Orders:** Upload a normal paid-order CSV. See actual order counts, "
            "time since the last order and spending summaries. This is historical "
            "analysis, not a prediction.\n\n"
            "**Research Lab:** For the supervisor or technical team. Review model "
            "comparison, experiments, evaluation methods and model settings.\n\n"
            "**Settings:** Understand score ranges, change the advanced yes/no "
            "cut-off for future analysis, or clear the current session."
        )
    with st.expander("Useful words explained in simple English"):
        st.markdown(
            "**Customer record:** One row of customer information at a given time. "
            "A customer can appear in more than one record.\n\n"
            "**Model risk score:** A number from 0 to 100 produced by the model. "
            "It is not yet a verified real-world probability.\n\n"
            "**Review priority:** A simple group based on the score. "
            "Higher means review earlier; it does not prove a customer will leave.\n\n"
             "**Churn definition:** The current code describes no purchase in four weeks. "
            "The thesis describes no interaction in the following month. These need verification.\n\n"
            "**Model explanation (SHAP):** A way to show which input patterns "
            "moved the model's estimate up or down. It does not show causation."
        )


elif page == "Store Orders (Preview)":
    title_block(
        "YOUR STORE'S ORDER HISTORY",
        "Understand your customers from an orders CSV",
        "Upload ordinary transaction data to see repeat buyers, spending and customers "
        "who have not placed an order recently. No AI setup or feature engineering needed for this view.",
    )
    section_block(
        "A simple view of your store's orders",
        "This page shows real past purchases. It does not guess what customers will buy next.",
    )
    card_grid([
        {"icon": "📂", "label": "1. Upload your orders",
         "description": "Choose your paid/completed orders CSV, or start with "
                        "the sample file below.", "tone": "purple"},
        {"icon": "🔗", "label": "2. Match your columns",
         "description": "Tell CHURNIQ which columns hold the buyer, order number, "
                        "date and optional amount.", "tone": "teal"},
        {"icon": "📊", "label": "3. Explore the results",
         "description": "See how often people ordered and who has not "
                        "bought recently. Download the summary.", "tone": "amber"},
    ])
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
        "Predict customer churn risk",
        "Upload a prepared customer-activity file and get an easy-to-read priority list. "
        "No AI knowledge is needed to interpret the results.",
    )
    section_block(
        "How churn prediction works",
        "The trained research model uses historical customer-activity data, not ordinary order rows.",
    )
    card_grid([
        {"icon": "📄", "label": "Choose prepared customer data",
         "description": "Upload a file with the same activity fields used "
                        "when training the REES46 model.", "tone": "purple"},
        {"icon": "🚩", "label": "Get clear review groups",
         "description": "CHURNIQ shows model scores and sorts records "
                        "into three easy-to-read priority groups.", "tone": "rose"},
        {"icon": "🧩", "label": "Understand each result",
         "description": "Open a customer to see which shopping signals "
                        "affected the score and what to review.", "tone": "teal"},
    ])
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
            section_block(
                "Your customer results",
                "The highest model scores appear first. "
                "These are churn risk estimates from the existing four-week no-purchase model.",
            )
            show_customer_table(result, limit=20)
            st.button(
                "Review the full customer list →",
                on_click=go_to, args=("Customer List",),
                key="check_to_customer_list", type="primary",
            )
            risk_note()


elif page == "Customer List":
    title_block(
        "CUSTOMER PRIORITIES",
        "Your churn risk customer list",
        "Search customers, focus on an attention level, and export a list for your team.",
    )
    section_block(
        "Your customer workspace",
        "Filter results to focus on the right customers. "
        "These are review groups, not promises about future purchases.",
    )
    card_grid([
        {"icon": "🔎", "label": "Find a customer",
         "description": "Search by customer ID or show only the priority "
                        "levels you want to review.", "tone": "purple"},
        {"icon": "🧠", "label": "Understand their score",
         "description": "Choose a customer and open the explanation page "
                        "to see model signals.", "tone": "teal"},
        {"icon": "⬇️", "label": "Share a review list",
         "description": "Download the filtered table as a CSV for "
                        "your team's manual review.", "tone": "amber"},
    ])
    result, _ = predictions_or_help()
    if result is not None:
        display = display_predictions(result)
        section_block("Find the customers you want to review")
        chosen_levels = st.multiselect(
            "Show customers marked",
            ["Needs attention", "Keep an eye on", "Lower priority"],
            default=["Needs attention", "Keep an eye on", "Lower priority"],
        )
        keyword = st.text_input("Search customer ID", placeholder="Enter any part of the customer ID")
        chosen = display[display["Priority"].isin(chosen_levels)]
        if keyword:
            chosen = chosen[chosen["Customer ID"].str.contains(keyword, case=False, regex=False)]
        chosen = chosen.sort_values("Churn risk score (not calibrated)", ascending=False)
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
        "Why did the model flag this customer for churn risk?",
        "Use SHAP to explore which historical shopping patterns influenced the churn "
        "risk estimate. These are model signals, not proven reasons why someone leaves.",
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
        col1.metric("Churn risk score", f"{selected_row['predicted_no_purchase_risk_pct']:.1f} / 100")
        col2.metric("Review priority", display_priority(selected_row["risk_band"]))
        risk_note()
        helpful_note(
            "Reading this page:",
            "The model checks historical shopping patterns and shows which ones pushed "
            "this estimate up or down. These patterns are clues, not proven causes."
        )
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
                "prove why a customer might churn or guarantee a successful follow-up."
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
        "Plan actions for customers at churn risk",
        "Use these suggestions as a starting point for your retention team, "
        "not as automatic marketing instructions.",
    )
    section_block(
        "From model score to human action",
        "Always check a person's real order history before making any decision.",
    )
    card_grid([
        {"icon": "🔎", "label": "Check the customer",
         "description": "Look at recent purchases, visits and any service problems. "
                        "A model score does not explain the whole story.", "tone": "purple"},
        {"icon": "💬", "label": "Choose a suitable next step",
         "description": "A reminder, product update or service follow-up may be "
                        "worth considering, if appropriate.", "tone": "amber"},
        {"icon": "🔒", "label": "Respect the customer",
         "description": "Follow consent and privacy rules. "
                        "CHURNIQ does not send messages for you.", "tone": "teal"},
    ])
    result, _ = predictions_or_help()
    if result is not None:
        high = result[result["risk_band"] == "High"]
        watch = result[result["risk_band"] == "Moderate"]
        col1, col2 = st.columns(2)
        col1.metric("Review first", len(high))
        col2.metric("Keep an eye on", len(watch))
        section_block(
            "Suggested checks for each review group",
            "These ideas are for your team to consider, not automated campaigns.",
        )
        card_grid([
            {"icon": "🔴", "label": "Needs attention — score 70+",
             "description": suggested_review_action("High"), "tone": "rose"},
            {"icon": "🟠", "label": "Keep an eye on — score 40–69",
             "description": suggested_review_action("Moderate"), "tone": "amber"},
            {"icon": "🟢", "label": "Lower priority — under 40",
             "description": suggested_review_action("Low"), "tone": "teal"},
        ])
        helpful_note(
            "Before contacting anyone:",
            "Check whether contact is appropriate, respect marketing consent "
            "and privacy requirements, and look for any recent order or support issue."
        )
        section_block(
            "Download a manual review shortlist",
            "This list contains customer IDs and model scores only; "
            "it does not send emails or promotions.",
        )
        priority_table = display_predictions(result[result["risk_band"].isin(["High", "Moderate"])])
        priority_table = priority_table.sort_values("Churn risk score (not calibrated)", ascending=False)
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
        "XGBoost results and SHAP research evidence",
        "The business pages keep things simple. This area preserves the full "
        "model comparison, experiment results and technical settings for your thesis.",
    )
    st.info("Thesis focus: " + PROJECT_TITLE)
    st.warning(TARGET_RECONCILIATION_NOTE)
    section_block(
        "What do these technical sections mean?",
        "These research results help explain how CHURNIQ was evaluated. "
        "They are separate from the everyday business workflow.",
    )
    card_grid([
        {"icon": "📊", "label": "Model comparison",
         "description": "See how different AI methods performed when checking "
                        "historical customer outcomes.", "tone": "purple"},
        {"icon": "🎯", "label": "Calibration study",
         "description": "Check how closely model scores match observed outcomes "
                        "in a separate research experiment.", "tone": "teal"},
        {"icon": "🕒", "label": "Future-period test",
         "description": "Test a separately trained model on a later period "
                        "to see how it handles changing customer behaviour.", "tone": "amber"},
        {"icon": "⚙️", "label": "Technical details",
         "description": "Review the input fields, training setup and settings "
                        "used by the loaded model.", "tone": "rose"},
    ], columns=4)
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
            st.write("**Model file:**", MODEL_PATH.name)
            st.write("**Model experiment:**", model_artifact_label(str(MODEL_PATH)))
            st.write("**Thesis churn definition:**", THESIS_CHURN_DEFINITION)
            st.write("**Current code target definition:**", report.get("project_target_definition", MODEL_CHURN_DEFINITION))
            st.write("**Outcome used by saved model:**", report.get("project_target_definition", MODEL_CHURN_DEFINITION))
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
    section_block(
        "Simple rules for reading results",
        "These settings help your team understand scores. "
        "They do not improve or retrain the AI model.",
    )
    card_grid([
        {"icon": "🚩", "label": "Priority levels",
         "description": "CHURNIQ groups scores into three demonstration ranges "
                        "so your team knows where to start.", "tone": "rose"},
        {"icon": "🎚️", "label": "Advanced threshold",
         "description": "You can change the yes/no cut-off used on your "
                        "next prediction run; review groups stay the same.", "tone": "purple"},
        {"icon": "🔒", "label": "Session data",
         "description": "Clear previously analysed customer results "
                        "from your current session.", "tone": "teal"},
    ])
    st.subheader("How customer priorities are assigned")
    st.write(
        "**Needs attention:** score of 70–100.  "
        "**Keep an eye on:** score of 40–69.9.  "
        "**Lower priority:** below 40."
    )
    st.caption("These are demonstration grouping rules, not calibrated probabilities or proven campaign thresholds.")
    with st.expander("Advanced: model classification threshold"):
        value = st.slider(
            "Threshold for the model's yes/no churn-risk flag",
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
    helpful_note(
        "About your privacy:",
        "This research prototype processes files in your running Streamlit session. "
        "Do not upload private customer information to public demos, "
        "and do not commit customer files to GitHub."
    )


st.divider()
st.caption(
    "CHURNIQ · UiTM research prototype · No customer outreach is automated · "
    "Scores are uncalibrated model estimates · Results are not proof of causation"
)
