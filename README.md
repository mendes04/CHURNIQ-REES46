# CHURNIQ — Customer Retention Intelligence for Online Stores

Research prototype for the thesis **Explainable Customer Churn Prediction in E-Commerce Using XGBoost and SHAP**. The default Streamlit interface is designed for e-commerce operators, while Research Lab preserves the technical evidence needed by academic examiners.

## Research target

The proposal defines churn as **no purchase during the following four weeks** (label `1`), as opposed to a purchase during those four weeks (label `0`). Prediction requires historical behavioural features, never the future target.

## Architecture

- `app.py` — Streamlit business dashboard and advanced research views
- `churniq/data.py` — validation, leakage-aware feature filtering and input schema
- `churniq/model.py` — preprocessing, Logistic Regression, Random Forest, default/tuned XGBoost, evaluation and saved pipeline
- `churniq/explain.py` — SHAP global and individual explanations
- `scripts/train_model.py` — reproducible offline experiment
- `scripts/audit_rees46.py` — safe aggregate metadata audit
- `scripts/make_demo_data.py` — synthetic data generator for testing only
- `tests/test_pipeline.py` — regression tests
- `EXHIBITION_PLAYBOOK.md` and `THESIS_CHAPTERS_4_5.md` — demonstration and academic evidence guides

**Current status:** The end-to-end workflow has been tested locally on REES46, including grouped customer holdout, four-model comparison, XGBoost tuning, individual/global SHAP, separate temporal evaluation and an exploratory sigmoid-calibration experiment. Model artifacts and customer-level records stay local; the GitHub repository contains code and tests, not your trained model or raw CSV.

## Open the e-commerce dashboard (after local training)

1. In GitHub Desktop, **Fetch origin → Pull origin**.
2. In your activated environment, run `python -m pytest -q`.
3. Ensure `artifacts/churniq_rees46.joblib` still contains the model you tested locally. Pulling source code does not remove your ignored model files.
4. Start it with `python -m streamlit run app.py` and open <http://localhost:8501>.

The business navigation is:

- **Business Home** — simple attention-level summary for currently analysed customer records.
- **Store Orders (Preview)** — upload an ordinary paid/completed orders CSV, inspect recency, sales and repeat purchasing, export an activity table, and review whether enough historical transactions exist to prepare four-week supervised learning examples. This is **descriptive and preparatory**, not a trained prediction system.
- **Check Customers** — upload a *prepared* behaviour-feature CSV or use the local historical example.
- **Customer List** — search, filter and export review-priority records.
- **Why Customers Are Flagged** — readable purchasing/browsing feature names, with technical SHAP scores in an expander.
- **Follow-up Planner** — human-review options, not automated or guaranteed retention outcomes.
- **Research Lab** — scientific comparison, calibration-study report, future-period evaluation and full feature details.
- **Settings** — explained demo risk bands, an advanced decision threshold and session cleanup.

**Important business limitation:** The existing REES46 XGBoost model still requires precomputed behavioural features in its original training schema (approximately 265 available original features). Ordinary Shopify/WooCommerce order exports **can now be analysed descriptively** on the Store Orders page, but **cannot yet be used to produce XGBoost four-week risk predictions**. Real predictive support would require a matching feature-engineering pipeline or a separate appropriately trained and evaluated orders-only model, as well as permission/privacy review. CHURNIQ does not claim an automated platform integration exists.

### Store Orders example

Click **Store Orders (Preview) → Download sample order CSV**, then upload that sample. Column mapping should automatically identify `customer_id`, `order_id`, `order_date` and `order_total`. Check each mapping before analysis. If an export has one row per item and repeats the full order total on each row, choose **Order total (may repeat for each item)** to avoid double counting. If it contains each item's individual value, choose **One amount per line item (add together)** instead.

The date cutoff defaults to the latest order in the file. Labels such as **No order in 30+ days** describe observed ordering history as of that date; they are **not predictions**.

The **Is this order history ready for AI predictions?** section applies clearly labelled, illustrative screening criteria: at least 12 weeks between earliest and latest order, 1,000 orders, 200 customers, three historical snapshots, and 50 labelled examples of each outcome. These are pragmatic sanity checks, **not statistically certified minimums**. The four-order example correctly fails. Candidate features are computed strictly before each retrospective snapshot; its target checks purchases during the 28 days following that snapshot. It does not train a model, establish sufficient future coverage, or transfer the existing REES46 pipeline to store orders. For trustworthy order-only prediction, a new modelling and chronological validation experiment must be designed and trained on richer, complete historical orders. The analyser does not automatically exclude refunded, cancelled or test orders or handle mixed currencies. Clean the export first. Store data is processed in your local Streamlit process and must not be committed to GitHub.

To generate a historical, held-out example file from the *same original training CSV*:

```powershell
python -m scripts.export_holdout_cohort --data "data/raw/rees46.csv" --model "artifacts/churniq_rees46.joblib" --output "data/sample/rees46_heldout_cohort.csv" --count 100 --period 0
```

The **Calibration study** and **Future-period test** tabs read local JSON reports, if present, from `data/reports/rees46_calibration_full.json` and `data/reports/rees46_temporal_full.json`. They describe *separate experiments* and do not change the model serving dashboard predictions. The displayed risk scores are not yet deployed as calibrated real-world probabilities.

## Install on Windows

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Place the *derived customer-level* REES46 CSV locally under `data/raw/` and **never commit customer-level records to GitHub**.

Inspect target/feature names and label definition before training.

```powershell
python scripts/audit_rees46.py --file "data/raw/YOUR_FILE.csv" --target "YOUR_TARGET_COLUMN"
python -m scripts.train_model --data "data/raw/YOUR_FILE.csv" --target "YOUR_TARGET_COLUMN"
streamlit run app.py
```

Run tests:

```powershell
python -m pytest -q
```

## Data and model safety

The target must not be present during inference. Review possible time leakage and repeated-customer overlap. Keep preprocessing and SMOTE inside CV training folds. The held-out test set must remain untouched until evaluation. Group-splitting by `user_id` is now used when an identifier is supplied. This is not a substitute for an additional forward-in-time robustness evaluation using `time_step`. Never load an untrusted joblib/pickle file. Do not claim synthetic results are actual REES46 results.

## Source

Derived Kaggle dataset: https://www.kaggle.com/datasets/fridrichmrtn/e-commerce-churn-dataset-rees46

This prototype does **not** establish real-world probability calibration, causal drivers, production privacy compliance, or retention-campaign outcomes.
