import numpy as np
import pandas as pd
import pytest
from churniq.data import prepare_training_data, validate_prediction_data
from churniq.model import train, predict
from churniq.explain import explain,top_local
from scripts.make_demo_data import generate


def test_train_infer_explain():
    df=generate(n=220)
    artifact=train(df,"churn","customer_id",fast=True,cv=2,max_features=6)
    assert len(artifact["report"]["scores"])==4
    assert "customer_id" not in artifact["features"]
    assert "churn" not in artifact["features"]
    batch=df.drop(columns=["churn"]).head(8)
    out=predict(artifact,batch,id_column="customer_id")
    assert len(out)==8
    assert out["predicted_inactivity_risk"].between(0,1).all()
    explanation=explain(artifact,batch.head(2))
    assert explanation["values"].shape[0]==2
    assert len(top_local(explanation,0))>0


def test_missing_required_feature_is_rejected():
    df=generate(n=200)
    artifact=train(df,"churn","customer_id",fast=True,cv=2,max_features=6)
    missing=df.drop(columns=["churn","total_sessions"])
    with pytest.raises(ValueError,match="Required input columns missing"):
        predict(artifact,missing)


def test_target_and_identifier_excluded():
    data=generate(n=100)
    x,y,dropped=prepare_training_data(data,"churn","customer_id")
    assert "churn" not in x and "customer_id" not in x
    assert set(y.unique())=={0,1}
