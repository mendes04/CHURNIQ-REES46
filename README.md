# CHURNIQ — Explainable E-Commerce Customer Inactivity Prediction

Research prototype for the thesis **Explainable Customer Churn Prediction in E-Commerce Using XGBoost and SHAP**.

## Research target

The proposal defines churn as **no interaction during the subsequent month** (label `1`), as opposed to return during that month (label `0`). Prediction requires historical behavioural features, never the future target.

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

**Repository setup status:** The initial runnable source is committed. The GitHub-hosted `app.py` is a streamlined implementation of the exhibition interface; the earlier CHURNIQ ZIP contains a more extensively styled variant you can choose to integrate later. No real REES46 model has yet been trained here.

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

The target must not be present during inference. Review possible time leakage and repeated-customer overlap. Keep preprocessing and SMOTE inside CV training folds. The held-out test set must remain untouched until evaluation. Never load an untrusted joblib/pickle file. Do not claim synthetic results are actual REES46 results.

## Source

Derived Kaggle dataset: https://www.kaggle.com/datasets/fridrichmrtn/e-commerce-churn-dataset-rees46

This prototype does **not** establish real-world probability calibration, causal drivers, production privacy compliance, or retention-campaign outcomes.
