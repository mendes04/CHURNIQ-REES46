"""Data loading, schema inspection, and leakage-aware feature preparation."""
from __future__ import annotations

import pandas as pd
import numpy as np

TARGET_ALIASES = ("churn", "target_class", "target", "label", "is_churn", "churned")
ID_ALIASES = ("visitorid", "customer_id", "customerid", "user_id", "userid", "id", "row_id", "rowid", "record_id")
DANGEROUS_TOKENS = ("future_", "next_month", "next_30", "subsequent", "target_", "label_", "churn_probability", "predicted_")


def load_csv(source) -> pd.DataFrame:
    df = pd.read_csv(source, low_memory=False)
    if df.empty:
        raise ValueError("CSV is empty.")
    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")
    return df


def suggest_column(columns, aliases):
    mapping = {c.lower().replace(" ", "_"): c for c in columns}
    for a in aliases:
        if a in mapping:
            return mapping[a]
    return None


def validate_target(df: pd.DataFrame, target: str) -> pd.Series:
    if target not in df:
        raise ValueError(f"Target column '{target}' not found.")
    if df[target].isna().any():
        raise ValueError("Training target has missing labels: remove or resolve these records first.")
    y = pd.to_numeric(df[target], errors="coerce")
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError("The target must be binary: 1 = no purchase in the following four weeks; 0 = purchased.")
    if y.nunique() != 2 or y.value_counts().min() < 5:
        raise ValueError("Both classes must contain at least five records.")
    return y.astype(int)


def prepare_training_data(df: pd.DataFrame, target: str, identifier: str | None = None):
    y = validate_target(df, target)
    exclude = {target}
    if identifier and identifier in df:
        exclude.add(identifier)
    elif identifier:
        raise ValueError(f"Identifier column '{identifier}' not present")
    for col in df:
        name = col.lower().replace(" ", "_")
        if col != target and (name in TARGET_ALIASES or "churn" in name or "label" in name or "target" in name):
            exclude.add(col)
        if col != identifier and name in ID_ALIASES:
            exclude.add(col)
        if any(part in name for part in DANGEROUS_TOKENS) or name == "time_step":
            exclude.add(col)
    x = df.drop(columns=sorted(exclude)).copy()
    dropped = list(sorted(exclude - {target}))
    for col in x.select_dtypes(include=["object", "string"]).columns:
        parsed = pd.to_numeric(x[col], errors="coerce")
        if parsed.notna().sum() >= max(3, int(0.95 * x[col].notna().sum())):
            x[col] = parsed
    for col in list(x.columns):
        s = x[col]
        if s.isna().all() or s.nunique(dropna=True) <= 1:
            x = x.drop(columns=col)
            dropped.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            x = x.drop(columns=col)
            dropped.append(col)
        elif (s.dtype == "object" or isinstance(s.dtype, pd.StringDtype)) and s.nunique(dropna=True) > 30:
            x = x.drop(columns=col)
            dropped.append(col)
    if x.shape[1] < 2:
        raise ValueError("Insufficient usable features after removing labels/IDs. Verify the dataset schema.")
    return x, y, dropped


def validate_prediction_data(df: pd.DataFrame, features: list[str], numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError("Required input columns missing: " + ", ".join(missing[:16]) + (" ..." if len(missing) > 16 else ""))
    x = df.loc[:, features].copy()
    for col in numeric:
        raw = x[col]
        value = pd.to_numeric(raw, errors="coerce")
        malformed = raw.notna() & value.isna()
        if malformed.any():
            raise ValueError(f"Column '{col}' has {int(malformed.sum())} values that are not numeric.")
        x[col] = value
    for col in categorical:
        x[col] = x[col].astype("object").where(x[col].notna(), np.nan)
    if x.isna().all(axis=1).any():
        raise ValueError("One or more customers have no usable behavioural values.")
    if x[numeric].replace([np.inf, -np.inf], np.nan).isna().any().any():
        if np.isinf(x[numeric].to_numpy(dtype=float)).any():
            raise ValueError("Numeric columns contain infinity.")
    return x


REES46_DESCRIPTIONS = {
    "ses_rec": "Days since last session",
    "ses_rec_avg": "Average time between sessions",
    "ses_rec_sd": "Variation in session gaps",
    "user_rec": "Time since first observed activity",
    "ses_n": "Total sessions",
    "int_n": "Total interactions",
    "tran_n": "Total purchases",
    "rev_sum": "Total purchase revenue",
    "ses_len_avg": "Average session length",
    "ses_wknd_r": "Share of weekend sessions",
    "time_to_int": "Time to first interaction",
    "time_to_tran": "Time to first purchase",
    "int_itm_n_avg": "Average unique items interacted with",
    "int_cat_n_avg": "Average categories interacted with",
}


def friendly_name(name: str) -> str:
    s = name.replace("num__", "").replace("cat__", "")
    if s in REES46_DESCRIPTIONS:
        return REES46_DESCRIPTIONS[s]
    return s.replace("_", " ").capitalize()
