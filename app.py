"""CHURNIQ Streamlit dashboard: thesis-aligned next-month inactivity predictions."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from churniq.data import load_csv, suggest_column, TARGET_ALIASES, ID_ALIASES
from churniq.model import load, predict, train, save
from churniq.explain import explain, top_local, top_global

st.set_page_config(page_title="CHURNIQ | Customer Intelligence", page_icon="◈", layout="wide")
st.markdown("""
<style>
[data-testid="stApp"]{background:#0b1020;color:#ecedfa}
[data-testid="stSidebar"]{background:#11192e}
h1,h2,h3{letter-spacing:-.03em}
[data-testid="stMetric"]{background:#161f36;border:1px solid #293754;border-radius:15px;padding:18px}
.stButton>button[kind="primary"]{background:#7567f5;color:white;border:0;border-radius:10px}
</style>""", unsafe_allow_html=True)

MODEL_PATH = Path(__file__).parent / "artifacts" / "churniq_rees46.joblib"


def artifact():
    if "model_bundle" not in st.session_state:
        try:
            st.session_state.model_bundle = load(MODEL_PATH) if MODEL_PATH.exists() else None
        except Exception as error:
            st.error(f"Cannot load locally trained model: {error}")
            st.session_state.model_bundle = None
    return st.session_state.model_bundle


def source():
    return st.session_state.get("predictions"), st.session_state.get("customers")


def require_predictions():
    result, raw = source()
    if result is None:
        st.info("Start from Upload & Predict to analyse new customer records.")
    return result, raw


bundle = artifact()
st.sidebar.title("◈ CHURNIQ")
st.sidebar.caption("EXPLAINABLE CUSTOMER INTELLIGENCE")
st.sidebar.success("Trained model loaded") if bundle else st.sidebar.warning("Model not trained")
page = st.sidebar.radio("Navigation", [
    "Overview", "Upload & Predict", "Customer Explorer",
    "Explainable AI", "Retention Planner", "Research Lab", "Settings"
])
st.sidebar.caption("Research prototype · REES46 · UiTM")

if page == "Overview":
    st.title("Understand customer inactivity before it happens")
    st.caption("XGBoost + SHAP · Target: no platform interaction during the following month")
    if bundle is None:
        st.info("Train a model on your actual labelled REES46 dataset first, then upload an unlabelled cohort.")
    else:
        report=bundle["report"]
        st.caption(f"Active: {report['model_used']} · Training records: {report['source_rows']:,}")
        result, raw = require_predictions()
        if result is not None:
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Customers analysed",f"{len(result):,}")
            c2.metric("High-risk customers",f"{(result.risk_band=='High').sum():,}")
            c3.metric("Moderate-risk customers",f"{(result.risk_band=='Moderate').sum():,}")
            c4.metric("Average risk score",f"{result.predicted_inactivity_risk.mean():.1%}")
            a,b=st.columns(2)
            with a:
                distribution=result.risk_band.value_counts().reindex(["Low","Moderate","High"],fill_value=0)
                st.plotly_chart(px.pie(names=distribution.index,values=distribution.values,hole=.65,
                                    color=distribution.index,color_discrete_map={"Low":"#1ed6b0","Moderate":"#f8bb6e","High":"#ff6d91"}),use_container_width=True)
            with b:
                st.subheader("Highest-risk customers")
                st.dataframe(result.nlargest(8,"predicted_inactivity_risk"),hide_index=True,use_container_width=True)
        st.caption("Scores are estimates, not guarantees. These are not predictions of permanent customer loss.")

elif page == "Upload & Predict":
    st.title("Upload & Predict")
    if bundle is None:
        st.error("A trained and trusted model artifact is required.")
    else:
        upload=st.file_uploader("Select customer-level features CSV (no churn label needed)",type="csv")
        st.caption(f"Model expects {len(bundle['features'])} feature columns with training-compatible definitions.")
        with st.expander("Required features"):
            st.write(bundle["features"])
        if upload is not None:
            try:
                data=load_csv(upload)
                st.dataframe(data.head(),hide_index=True,use_container_width=True)
                missing=[c for c in bundle["features"] if c not in data.columns]
                if missing:
                    st.error("Missing required features: " + ", ".join(missing[:20]))
                else:
                    ids=["Generate row IDs"]+[c for c in data.columns if c not in bundle["features"]]
                    chosen=st.selectbox("Identifier",ids)
                    if st.button("Run real model predictions",type="primary"):
                        result=predict(bundle,data,threshold=st.session_state.get("threshold",.5),
                                      id_column=None if chosen==ids[0] else chosen)
                        st.session_state.customers=data
                        st.session_state.predictions=result
                        st.success(f"Predictions generated for {len(result):,} customers")
                        st.dataframe(result.head(20),hide_index=True,use_container_width=True)
            except Exception as error:
                st.error(str(error))

elif page == "Customer Explorer":
    st.title("Customer Explorer")
    result,raw=require_predictions()
    if result is not None:
        band=st.multiselect("Risk band",["High","Moderate","Low"],default=["High","Moderate","Low"])
        query=st.text_input("Search customer ID")
        filtered=result[result.risk_band.isin(band)]
        if query:
            filtered=filtered[filtered.customer_id.str.contains(query,case=False,regex=False)]
        st.dataframe(filtered.sort_values("predicted_inactivity_risk",ascending=False),
                     hide_index=True,use_container_width=True)
        st.download_button("Export predictions",result.to_csv(index=False),file_name="churniq_predictions.csv",mime="text/csv")

elif page == "Explainable AI":
    st.title("Why did the model give this prediction?")
    result,raw=require_predictions()
    if bundle is not None and result is not None:
        st.info("SHAP explains contribution to the XGBoost model score, not the cause of a customer's actual behaviour.")
        choice=st.selectbox("Select customer",range(len(result)),
                            format_func=lambda i:f"{result.iloc[i].customer_id} — {result.iloc[i].predicted_inactivity_risk_pct:.1f}%")
        try:
            sample=raw.iloc[[choice]]
            detail=explain(bundle,sample,max_rows=1)
            local=top_local(detail,0)
            local["name"]=local.feature.str.replace("num__","",regex=False).str.replace("cat__","",regex=False)
            fig=px.bar(local.sort_values("contribution_log_odds"),x="contribution_log_odds",y="name",orientation="h",
                       color="contribution_log_odds",color_continuous_scale="RdBu_r",
                       labels={"contribution_log_odds":"SHAP contribution (log-odds)","name":"Behavioural feature"})
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Positive SHAP values increase the model's inactivity risk score; negative values decrease it.")
            with st.expander("Global feature importance (sample of uploaded customers)"):
                global_exp=explain(bundle,raw,max_rows=min(150,len(raw)))
                importance=top_global(global_exp)
                st.plotly_chart(px.bar(importance.sort_values("mean_abs_shap_log_odds"),x="mean_abs_shap_log_odds",y="feature",orientation="h"),
                                use_container_width=True)
        except Exception as error:
            st.error(str(error))

elif page == "Retention Planner":
    st.title("Customers for retention review")
    result,_=require_predictions()
    if result is not None:
        priority=result[result.risk_band=="High"].sort_values("predicted_inactivity_risk",ascending=False)
        st.metric("High-risk customers",len(priority))
        st.dataframe(priority,hide_index=True,use_container_width=True)
        st.download_button("Download shortlist",priority.to_csv(index=False),file_name="churniq_review.csv",mime="text/csv")
        st.info("Review SHAP explanations and customer context before considering outreach. No automated intervention occurs.")

elif page == "Research Lab":
    st.title("Research Lab · Examiner View")
    tab1,tab2=st.tabs(["Evaluation","Train model"])
    with tab1:
        if bundle is None:
            st.info("No trained model yet.")
        else:
            report=bundle["report"]
            st.dataframe(pd.DataFrame(report["scores"]).T,use_container_width=True)
            fig=px.line()
            for name,values in report["roc_curves"].items():
                fig.add_scatter(x=values["fpr"],y=values["tpr"],name=name,mode="lines")
            fig.update_layout(xaxis_title="False positive rate",yaxis_title="True positive rate")
            st.plotly_chart(fig,use_container_width=True)
            st.json({"best_params":report["best_params"],"split":report["split"],
                     "selected_features":report["selected_transformed_features"]})
    with tab2:
        st.warning("Offline CLI training is preferred. Real REES46 modelling may be computationally expensive.")
        upload=st.file_uploader("Labelled training CSV",type="csv",key="train")
        if upload:
            try:
                df=load_csv(upload)
                target_guess=suggest_column(df.columns,TARGET_ALIASES)
                target=st.selectbox("Next-month target column",df.columns.tolist(),
                                    index=df.columns.get_loc(target_guess) if target_guess else 0)
                ident=st.selectbox("Customer ID",["None"]+[c for c in df.columns if c!=target])
                fast=st.checkbox("Fast smoke test (NOT thesis experiment)",value=False)
                if st.button("Train four models",type="primary"):
                    trained=train(df,target,None if ident=="None" else ident,fast=fast)
                    save(trained,MODEL_PATH)
                    st.session_state.model_bundle=trained
                    st.session_state.pop("predictions",None)
                    st.session_state.pop("customers",None)
                    st.success("Model trained and saved. View Evaluation.")
            except Exception as error:
                st.error(str(error))

elif page == "Settings":
    st.title("System Settings")
    threshold=st.slider("Classification threshold",0.05,0.95,float(st.session_state.get("threshold",0.5)),0.05)
    st.session_state.threshold=threshold
    st.caption("Fixed display risk bands: Low <40%, Moderate 40–69.9%, High ≥70%. Calibrate probabilities before interpreting literal chances.")
    if bundle:
        st.write("Expected model features")
        st.dataframe(pd.DataFrame({"name":bundle["features"]}),hide_index=True)
    if st.button("Clear customer session data"):
        st.session_state.pop("predictions",None)
        st.session_state.pop("customers",None)
        st.success("Customer records cleared from current session.")

st.divider()
st.caption("CHURNIQ · Thesis prototype · No fabricated model metrics · SHAP is attribution, not causation")
