"""Run the IN6227 classification experiment."""
import argparse
import hashlib
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

from data import TARGET, audit, load_data, save_json
from modeling import THRESHOLD, bootstrap_ap_difference, evaluate_models, fit_models

SEED = 6227
FOLDS = 5


def run(data_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("error", category=ConvergenceWarning)
    raw_train = load_data(data_dir / "train.csv")
    train = raw_train.dropna(subset=[TARGET])
    X, y = train.drop(columns=TARGET), train[TARGET].eq("yes").astype(int)
    if X.duplicated().any():
        raise ValueError("Duplicate training features: review grouped CV")
    folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y))
    fold_ids = np.empty(len(X), dtype=int)
    for number, (_, validation) in enumerate(folds, start=1):
        fold_ids[validation] = number
    pd.DataFrame({"csv_line": train.index + 2, "validation_fold": fold_ids}).to_csv(
        output / "cv_folds.csv", index=False)
    report = {
        "seed": SEED, "target": TARGET, "positive_label": "yes", "threshold": THRESHOLD,
        "cv_folds": FOLDS, "selection_metric": "average_precision", "class_weight": None,
        "numeric_columns": X.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": X.select_dtypes(exclude="number").columns.tolist(),
        "train_raw": audit(raw_train), "train_used": audit(train),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
    }
    models, report["models"] = fit_models(X, y, folds, SEED, output)
    report["selected_model"] = max(report["models"], key=lambda name: report["models"][name]["cv_ap_mean"])
    report["configuration_frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(output / "selection_before_test.json", report)

    # Read test data only after model selection.
    raw_test = load_data(data_dir / "test.csv")
    if list(raw_test.columns) != list(raw_train.columns):
        raise ValueError("Train/test schema mismatch")
    test = raw_test.dropna(subset=[TARGET])
    Xt, yt = test.drop(columns=TARGET), test[TARGET].eq("yes").astype(int).to_numpy()
    overlap = pd.util.hash_pandas_object(Xt, index=False).isin(pd.util.hash_pandas_object(X, index=False))
    if overlap.any():
        raise ValueError("Train/test feature overlap: review the split")
    report.update({
        "test_raw": audit(raw_test), "test_used": audit(test), "train_test_feature_overlap_rows": 0,
        "unseen_test_categories": {c: sorted(set(Xt[c].dropna()) - set(X[c].dropna()))
                                   for c in report["categorical_columns"]},
        "input_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                         for name in ["train.csv", "test.csv"]},
    })
    results, report["majority_baseline"], scores = evaluate_models(models, Xt, yt, y.mean(), output)
    for name, values in results.items():
        report["models"][name]["test"] = values
    report["bootstrap_ap_difference_rf_minus_lr"] = bootstrap_ap_difference(yt, scores, SEED)
    save_json(output / "experiment.json", report)
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.data_dir, args.output)
