# CHURNIQ thesis-to-implementation alignment audit

**Thesis:** Explainable Customer Churn Prediction in E-Commerce Using XGBoost and SHAP (Chapters 1–3). This document audits implementation consistency; it does not revise experimental outcomes.

## Already aligned

- REES46 customer behavioural data and grouped 80/20 holdout.
- Logistic Regression, Random Forest, default and tuned XGBoost comparison.
- Three-fold grouped cross-validation; SelectKBest feature selection; training-only SMOTE.
- Saved tuned XGBoost pipeline; Streamlit business dashboard; global and local SHAP.

## Differences requiring verification

| Thesis | Current code / evidence | Resolution |
|---|---|---|
| Chapter 3 Table 3.2: churn is **no customer interaction next month** | Training code describes target_event=1 as **no purchase in the following four weeks** | **High priority:** verify the original REES46 target construction and class mapping. Interacting without purchasing is possible. Never assume they are equivalent or relabel without verification. |
| Thesis states **112,611 records** | Saved training report records **112,610 rows** | Confirm exactly which source CSV and preprocessing were used; document the final analysed row count. |
| Chapter 3.5 refers to duplicate-row removal and outlier handling | Code checks duplicate column names and filters input fields but does not explicitly drop duplicate data rows or detect outliers | Run a documented audit or revise methods to say what was actually implemented. |
| Thesis describes SMOTE before feature selection | Implemented pipeline: imputation/encoding/scaling, SelectKBest(k=40), then SMOTE, then classifier | Align the methodology diagram/text with the executed order or run and document a fresh experiment. |
| Chapter 3.7 lists tuning of scale_pos_weight and subsample | Full search tunes max_depth, learning_rate, n_estimators; subsample fixed at 0.85; scale_pos_weight not searched | Correct method details or carry out a new controlled experiment. |
| Chapter 3.9–3.11 envisages dashboard training | Streamlit currently loads a saved model; reproducible model fitting occurs from the command line | Describe it accurately as a churn-risk inference and SHAP dashboard with a Research Lab. |
| General business platform support | Existing XGBoost model requires engineered REES46 features; normal order files are descriptive only | State input restrictions and do not claim live store prediction from arbitrary order exports. |
| SHAP interprets churn drivers | SHAP explains model attribution in log-odds, not proven causal reasons | Keep global/local visualisations and explain limitations. |

## UI alignment already implemented

- Main display uses customer churn prediction, churn risk score, churn risk categories and SHAP explanations.
- Full tuned XGBoost candidate is preferred if saved locally; otherwise an older demonstration model is loaded and labelled.
- The automatically selected example file matches the active model's source.
- Internal classification targets, fitted models, SMOTE, SHAP calculations, evaluation results, and original data have not been changed.
- Research Lab shows the unresolved thesis-versus-code target-definition discrepancy.

## Critical scientific boundary

The word *churn* on a dashboard does not prove that the classifier predicts *no interaction* when its code documents *no purchase*. Confirm the actual REES46 target_event provenance and encoding before treating the labels as identical. If the label definition changes, the model and evaluation may need to be rebuilt and documented. Do not present old scores as new experiment results.

## Files to check

- Training: churniq/data.py, churniq/model.py, scripts/train_model.py.
- UI: app.py, churniq/business.py, churniq/thesis_alignment.py.
- Main thesis candidate model (local only): artifacts/churniq_full_candidate.joblib.
- Prepared thesis example (local only): data/sample/rees46_chapter4_holdout.csv.
