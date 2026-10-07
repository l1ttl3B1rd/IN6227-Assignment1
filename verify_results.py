"""Check saved predictions using independent metric formulas."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def ranking_metrics(y, p):
    # Group tied scores before integrating the ranking curves.
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(np.diff(ps)), len(ps) - 1]
    tp = np.cumsum(ys)[ends]
    fp = ends + 1 - tp
    recall = tp / y.sum()
    ap = np.sum(np.diff(np.r_[0., recall]) * tp / (ends + 1))
    tpr = np.r_[0., recall]
    fpr = np.r_[0., fp / (len(y) - y.sum())]
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    return float(ap), float(auc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    r = json.loads((args.results / "experiment.json").read_text())
    d = pd.read_csv(args.results / "test_predictions.csv")
    y = d.y_true.to_numpy()
    assert len(y) == r["test_used"]["rows"]
    assert int(y.sum()) == r["test_used"]["label_counts"]["yes"]
    checks = {}
    for name, model in r["models"].items():
        slug = name.lower().replace(" ", "_")
        p = d[slug + "_prob_yes"].to_numpy()
        assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
        prediction = (p >= r["threshold"]).astype(int)
        assert np.array_equal(prediction, d[slug + "_pred"].to_numpy())
        tp = int(((y == 1) & (prediction == 1)).sum())
        tn = int(((y == 0) & (prediction == 0)).sum())
        fp = int(((y == 0) & (prediction == 1)).sum())
        fn = int(((y == 1) & (prediction == 0)).sum())
        ap, auc = ranking_metrics(y, p)
        values = {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                  "accuracy": (tn + tp) / len(y), "f1_yes": 2 * tp / (2 * tp + fp + fn),
                  "recall_yes": tp / (tp + fn), "precision_yes": tp / (tp + fp),
                  "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
                  "average_precision": ap, "roc_auc": auc}
        for key, value in values.items():
            assert abs(value - model["test"][key]) < 1e-12, (name, key)
        assert sum([tn, fp, fn, tp]) == len(y)
        checks[name] = {"independent_metrics": values, "status": "PASS"}
    frozen = json.loads((args.results / "selection_before_test.json").read_text())
    assert "test_raw" not in frozen
    for name in r["models"]:
        assert frozen["models"][name]["best_params"] == r["models"][name]["best_params"]
        assert "test" not in frozen["models"][name]
    assert r["train_test_feature_overlap_rows"] == 0
    folds = pd.read_csv(args.results / "cv_folds.csv")
    assert folds.csv_line.is_unique and len(folds) == r["train_used"]["rows"]
    assert sorted(folds.validation_fold.unique()) == [1, 2, 3, 4, 5]
    checks["split_and_selection"] = "PASS"
    (args.results / "verification.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
