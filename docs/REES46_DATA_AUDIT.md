# REES46 Dataset Audit — 23 October 2026

## Source and definition

Dataset: locally extracted `rees46.csv`; metadata exported by `scripts/audit_rees46.py`.
Original feature-generation code: https://github.com/fridrichmrtn/churn-modeling/blob/main/code/ingestion/customer-model/workhorse.py

`target_event=1` means **no purchase in the following four-week target period**; `target_event=0` means **at least one purchase in that period**. This is NOT equivalent to no website interaction, permanent churn, or switching to a competitor.

## Verified local data profile

- Total rows: **112,610**
- Dataset columns: **276**
- Unique customers (`user_id`): **21,605**
- Additional repeated-customer rows: **91,005**
- Duplicated (`user_id`, `time_step`) pairs: **0**
- Columns with missing entries (from aggregate audit): **0**
- Binary `target_event=1`: **36,130** (32.08%)
- Binary `target_event=0`: **76,480** (67.92%)

The aggregation has multiple observations per customer across seven time steps; records cannot be treated as independent customers.

| time_step | Purchased (0) | No purchase (1) | Rows | No-purchase rate |
|---:|---:|---:|---:|---:|
| 0 | 7,916 | 13,689 | 21,605 | 63.36% |
| 1 | 12,996 | 8,034 | 21,030 | 38.20% |
| 2 | 13,944 | 5,746 | 19,690 | 29.18% |
| 3 | 13,595 | 4,226 | 17,821 | 23.71% |
| 4 | 12,666 | 2,309 | 14,975 | 15.42% |
| 5 | 9,674 | 1,559 | 11,233 | 13.88% |
| 6 | 5,689 | 567 | 6,256 | 9.06% |
| **All** | **76,480** | **36,130** | **112,610** | **32.08%** |

## Modelling consequences

1. **Never use identifiers as predictors:** `row_id`, `user_id`.
2. **Remove alternative and future outcome fields from predictors:** `target_revenue`, `target_customer_value`, `target_customer_value_lag1`, `target_actual_profit`.
3. Treat `time_step` as a period/splitting index rather than a behaviour feature. Validate the chronological definition with original source generation.
4. The main implementation currently uses a customer-level group-safe train/test split plus group-safe cross-validation to prevent the same customer appearing across train and test. This does **not** establish out-of-time validity.
5. The class rate varies from **9.06%** to **63.36%** across periods. This could reflect cohort composition and/or time effects; evaluate each period separately and add an out-of-time holdout.
6. Fit imputation, feature selection and any SMOTE resampling **only on training folds**, never on the test set.
7. Evaluate probability calibration before reporting model scores as literal probabilities.
8. Report exact train/test sizes, selected features, confusion matrices, precision, recall, F1, ROC-AUC and per-period results. No results have yet been produced on the real file.

## Open questions before thesis conclusions

- Are the labels and explanatory feature windows temporally consistent with a four-week forecast for each time step?
- How well does an older-period model generalise to a newer period?
- Does class balancing improve minority-class F1/recall without harming precision or probability calibration?
- Can dominant SHAP factors be translated into understandable behaviour categories rather than opaque latent factors?

This document is a **data audit and design decision record**, not a trained-model evaluation.
