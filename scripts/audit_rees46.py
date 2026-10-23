"""REES46 dataset metadata audit. Exports aggregates, never customer rows."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

ALIASES = ("churn", "target_class", "target", "label", "is_churn", "churned")
RISK_TERMS = ("future", "next_month", "nextmonth", "after_", "target", "label", "churn", "outcome", "subsequent", "following", "post_")
IDENTIFIER_TERMS = ("customer_id", "visitorid", "visitor_id", "user_id", "email", "phone", "full_name", "session_id")


def native_number(x):
    if x is None or pd.isna(x):
        return None
    try:
        y = float(x)
        return y if math.isfinite(y) else None
    except (ValueError, TypeError):
        return None


def audit(file_path: Path, target: str | None, batch_size: int):
    if not file_path.is_file():
        raise FileNotFoundError(f"Cannot find file: {file_path}")
    if not file_path.name.lower().endswith((".csv", ".csv.gz", ".zip")):
        raise ValueError("Expected CSV, CSV.GZ or a ZIP containing one CSV")
    reader = pd.read_csv(file_path, chunksize=batch_size, low_memory=False)
    result = None
    target_counter = Counter()
    rows = 0
    for chunk in reader:
        if result is None:
            columns = [str(c) for c in chunk.columns]
            if target is None:
                target = next((c for c in columns if c.lower() in ALIASES), None)
            if target is not None and target not in columns:
                raise ValueError(f"Target '{target}' does not match a column. Found: {columns}")
            result = {
                "dataset_file": file_path.name,
                "row_count": 0,
                "column_count": len(columns),
                "target_column": target,
                "column_names": columns,
                "columns": {c: {"type_observed_in_first_chunk": str(chunk[c].dtype), "null_count": 0, "null_pct": 0.0} for c in columns},
                "potential_future_label_or_leakage_columns_review_manually": [c for c in columns if any(t in c.lower() for t in RISK_TERMS)],
                "potential_identifier_columns_review_manually": [c for c in columns if any(t in c.lower() for t in IDENTIFIER_TERMS)],
                "numeric_ranges": {},
                "target_distribution": {},
                "notes": ["Metadata/aggregates only: NO raw customer records exported.",
                          "Column checks do not prove observations precede the predicted next month."]
            }
        rows += len(chunk)
        for c in columns:
            result["columns"][c]["null_count"] += int(chunk[c].isna().sum())
            if is_numeric_dtype(chunk[c]) and c != target:
                valid = pd.to_numeric(chunk[c], errors="coerce").dropna()
                if not valid.empty:
                    lo = native_number(valid.min())
                    hi = native_number(valid.max())
                    old = result["numeric_ranges"].get(c)
                    result["numeric_ranges"][c] = {
                        "min": min(lo, old["min"]) if old and old["min"] is not None and lo is not None else lo,
                        "max": max(hi, old["max"]) if old and old["max"] is not None and hi is not None else hi
                    }
        if target:
            target_counter.update({str(k): int(v) for k, v in chunk[target].fillna("<MISSING>").astype(str).value_counts().items()})
    if result is None:
        raise ValueError("CSV has no data chunks.")
    result["row_count"] = rows
    for c in result["column_names"]:
        result["columns"][c]["null_pct"] = round(100 * result["columns"][c]["null_count"] / max(1, rows), 3)
    result["target_distribution"] = dict(target_counter.most_common(25))
    return result


def main():
    parser = argparse.ArgumentParser(description="Export aggregate REES46 metadata")
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--target", default=None)
    parser.add_argument("--output", type=Path, default=Path("rees46_profile.json"))
    parser.add_argument("--chunksize", type=int, default=25000)
    args = parser.parse_args()
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    report = audit(args.file, args.target, args.chunksize)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {args.output} | rows={report['row_count']:,} | columns={report['column_count']}")
    print("Review generated metadata before sharing.")


if __name__ == "__main__":
    main()
