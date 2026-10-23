"""CLI training — preferred over dashboard training for reproducible research runs.
Example:
python -m scripts.train_model --data data/REES46.csv --target target_class --id visitorid
"""
from __future__ import annotations
import argparse
from churniq.data import load_csv, suggest_column, TARGET_ALIASES, ID_ALIASES
from churniq.model import train, save


def main():
    p=argparse.ArgumentParser(description="Train CHURNIQ on labelled customer-level REES46 data")
    p.add_argument("--data",required=True,help="Path to your extracted Kaggle CSV")
    p.add_argument("--target",help="Actual binary next-month churn column name")
    p.add_argument("--id",help="Customer ID column (excluded from training)")
    p.add_argument("--output",default="artifacts/churniq_rees46.joblib")
    p.add_argument("--cv",type=int,default=3)
    p.add_argument("--features",type=int,default=40)
    p.add_argument("--fast",action="store_true",help="Small hyperparameter grid for smoke tests")
    a=p.parse_args()
    df=load_csv(a.data)
    target=a.target or suggest_column(df.columns,TARGET_ALIASES)
    identifier=a.id or suggest_column(df.columns,ID_ALIASES)
    if not target:raise ValueError("No obvious target found. Supply --target exactly as named in CSV.")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns)} | Target: {target} | ID: {identifier}")
    print("Confirm label 1 means no subsequent-month interaction and 0 means returned.")
    artifact=train(df,target,identifier,fast=a.fast,cv=a.cv,max_features=a.features)
    path=save(artifact,a.output)
    print(f"Model saved to: {path}")
    for model,metrics in artifact["report"]["scores"].items():
        print(f"{model:22s} F1={metrics['f1']:.3f} Recall={metrics['recall']:.3f} ROC-AUC={metrics['roc_auc']:.3f}")
    print(f"Best by held-out test F1 (comparison only): {artifact['report']['best_by_test_f1']}")


if __name__=="__main__":
    main()
