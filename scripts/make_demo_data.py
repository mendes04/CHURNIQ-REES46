"""Creates SYNTHETIC rows for pipeline testing, NOT actual REES46 research results.
Use only to test upload/prediction mechanics.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate(n=600, seed=42):
    rng=np.random.default_rng(seed)
    recency=rng.gamma(2.6,9,n).round(1)
    sessions=rng.poisson(14,n)
    purchases=np.minimum(sessions,rng.poisson(3,n))
    product_views=np.maximum(sessions,rng.poisson(44,n))
    spend=(purchases*rng.uniform(15,70,n)).round(2)
    cart=rng.poisson(6,n)
    category=rng.choice(["electronics","home","fashion"],n)
    logit=-1.4+0.045*recency-0.08*purchases-0.023*sessions+0.01*np.maximum(0,20-cart)
    prob=1/(1+np.exp(-logit))
    churn=(rng.uniform(0,1,n)<prob).astype(int)
    return pd.DataFrame({"customer_id":[f"SYN-{j:05d}" for j in range(n)],
                         "days_since_last_session":recency,"total_sessions":sessions,
                         "total_purchases":purchases,"product_views":product_views,
                         "cart_additions":cart,"total_spending":spend,
                         "preferred_category":category,"churn":churn})


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--n",type=int,default=600)
    a=ap.parse_args()
    df=generate(a.n)
    out=Path("sample_synthetic_training.csv")
    df.to_csv(out,index=False)
    df.drop(columns=["churn"]).head(60).to_csv("sample_synthetic_prediction.csv",index=False)
    print("Created SYNTHETIC example CSVs; not evidence of REES46 performance")
