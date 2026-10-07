"""Dataset loading and audit."""
import json
import numpy as np
import pandas as pd

TARGET = "label"

def save_json(path, obj):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj, indent=2, default=convert, allow_nan=False), encoding="utf-8")



def load_data(path):
    frame = pd.read_csv(path)
    for col in frame.select_dtypes(include="object"):
        frame[col] = frame[col].str.strip().replace("", np.nan)
    if TARGET not in frame:
        raise ValueError(f"Missing target column: {TARGET}")
    observed = set(frame[TARGET].dropna().unique())
    if not observed <= {"yes", "no"}:
        raise ValueError(f"Unexpected labels: {observed}")
    return frame



def audit(frame):
    features = frame.drop(columns=TARGET)
    numbers = features.select_dtypes(include="number")
    if np.isinf(numbers.to_numpy()).any():
        raise ValueError("Non-finite numeric inputs require an explicit cleaning policy")
    q1, q3 = numbers.quantile(.25), numbers.quantile(.75)
    outside = (numbers < q1 - 1.5 * (q3 - q1)) | (numbers > q3 + 1.5 * (q3 - q1))
    return {
        "rows": len(frame), "features": features.shape[1],
        "missing_labels": int(frame[TARGET].isna().sum()),
        "missing_features": int(features.isna().sum().sum()),
        "rows_with_missing_features": int(features.isna().any(axis=1).sum()),
        "missing_by_column": frame.isna().sum().to_dict(),
        "label_counts": frame[TARGET].dropna().value_counts().to_dict(),
        "full_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "numeric_summary": numbers.describe().to_dict(),
        "iqr_outlier_counts": outside.sum().to_dict(),
        "rows_with_any_iqr_outlier": int(outside.any(axis=1).sum()),
        "numeric_correlations": numbers.corr().to_dict(),
        "category_counts": {c: features[c].dropna().value_counts().to_dict()
                            for c in features.select_dtypes(include="object")},
    }
