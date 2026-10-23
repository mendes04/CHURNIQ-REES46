# CHURNIQ Exhibition Playbook

## Before the event
- Train **on the actual derived REES46 customer-level dataset** and validate its next-month target.
- Evaluate Logistic Regression, Random Forest, default XGBoost and tuned XGBoost on an untouched test set.
- Verify new unlabelled customer rows generate predictions; demonstrate individual and global SHAP.
- Rehearse on the same laptop, using a local/offline Streamlit installation.
- Use only synthetic or anonymised prediction records in public demonstrations.
- Bring printed thesis architecture, model comparison and clearly labelled real performance metrics.
- Keep a trusted offline copy of your trained pipeline, outside GitHub.

## Three-minute presentation
1. Explain the business problem and exact target: no purchase in the **following four weeks**.
2. Show the real model evaluation and what was optimised, without claiming tuned XGBoost necessarily wins.
3. Upload a compatible, unlabelled customer dataset; run predictions.
4. Search for a high-risk customer and show the actual local SHAP explanation.
5. Show global drivers, the retention review list and the export.
6. End with study limitations: SHAP is attribution, not causality; high risk does not guarantee churn.

## Final checks
- Run `python -m pytest -q`.
- Ensure no sample or synthetic scores appear as actual thesis findings.
- Confirm every exported row matches its displayed prediction.
- Keep the interface readable at exhibition monitor resolution.
- Avoid public hosting of identifiable customer data.
