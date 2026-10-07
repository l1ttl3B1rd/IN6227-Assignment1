"""Preprocessing, model selection and evaluation."""
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

THRESHOLD = 0.5

def make_pipeline(numeric, categorical, model, scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    preprocess = ColumnTransformer([
        ("numeric", Pipeline(num_steps), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], remainder="drop")
    return Pipeline([("preprocess", preprocess), ("model", model)])



def metrics(y, scores):
    prediction = (scores >= THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "average_precision": average_precision_score(y, scores),
        "roc_auc": roc_auc_score(y, scores),
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "precision_yes": precision_score(y, prediction, zero_division=0),
        "recall_yes": recall_score(y, prediction, zero_division=0),
        "f1_yes": f1_score(y, prediction, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_models(X, y, folds, seed, output):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    specifications = {
        "Logistic regression": (
            LogisticRegression(solver="lbfgs", max_iter=2000, tol=1e-4, random_state=seed),
            {"model__C": [0.1, 1.0, 10.0]}, True),
        "Random forest": (
            RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=2),
            {"model__max_depth": [12, None], "model__min_samples_leaf": [2, 10]}, False),
    }
    models, summaries = {}, {}
    for name, (estimator, grid, scale) in specifications.items():
        search = GridSearchCV(make_pipeline(numeric, categorical, estimator, scale), grid,
                              scoring="average_precision", cv=folds, n_jobs=1,
                              error_score="raise", return_train_score=True)
        started = time.perf_counter()
        with threadpool_limits(limits=2):
            search.fit(X, y)
        elapsed = time.perf_counter() - started
        best, scores = search.best_index_, search.cv_results_
        model = models[name] = search.best_estimator_
        summaries[name] = {
            "search_grid": grid, "best_params": search.best_params_,
            "cv_ap_mean": search.best_score_, "cv_ap_std": scores["std_test_score"][best],
            "cv_ap_folds": [scores[f"split{i}_test_score"][best] for i in range(len(folds))],
            "cv_training_ap_mean": scores["mean_train_score"][best],
            "search_and_refit_seconds": elapsed, "refit_seconds": search.refit_time_,
            "encoded_feature_count": len(model["preprocess"].get_feature_names_out()),
            "estimator_params": model["model"].get_params(),
        }
        if hasattr(model["model"], "n_iter_"):
            summaries[name]["refit_iterations"] = model["model"].n_iter_.tolist()
        pd.DataFrame(scores).to_csv(output / f"cv_{name.lower().replace(' ', '_')}.csv", index=False)
        print(f"{name}: {search.best_params_}; CV AP={search.best_score_:.6f}", flush=True)
    return models, summaries


def evaluate_models(models, X, y, train_prevalence, output):
    predictions = pd.DataFrame({"csv_line": X.index + 2, "y_true": y})
    scores, results = {}, {}
    for name, model in models.items():
        with threadpool_limits(limits=2):
            scores[name] = model.predict_proba(X)[:, 1]
        slug = name.lower().replace(" ", "_")
        predictions[slug + "_prob_yes"] = scores[name]
        predictions[slug + "_pred"] = (scores[name] >= THRESHOLD).astype(int)
        results[name] = metrics(y, scores[name])
    baseline = metrics(y, np.full(len(y), train_prevalence))
    rows = [{"model": name, **values} for name, values in results.items()]
    rows.append({"model": "Always no baseline", **baseline})
    predictions.to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "test_metrics.csv", index=False)
    return results, baseline, scores


def bootstrap_ap_difference(y, scores, seed, repeats=1000):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == value) for value in [0, 1]]
    left, right = scores["Random forest"], scores["Logistic regression"]
    differences = []
    for _ in range(repeats):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        differences.append(average_precision_score(y[indices], left[indices]) -
                           average_precision_score(y[indices], right[indices]))
    return {
        "replicates": repeats, "method": "paired stratified percentile bootstrap; fixed fitted models",
        "point": average_precision_score(y, left) - average_precision_score(y, right),
        "ci95": np.quantile(differences, [.025, .975]).tolist(),
    }
