from __future__ import annotations

import argparse
from joblib import dump
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .evaluate import compute_metrics, plot_curves, plot_lift, classification_metrics, best_threshold_by_ks
from .models import build_hist_gb, build_random_forest, build_xgboost
from .utils import ensure_dir, load_train, save_json


def fit_predict(model, X_tr, y_tr, X_va, y_va):
    # Special-case XGBoost: early stopping on transformed matrices
    if model.named_steps["model"].__class__.__name__.lower().startswith("xgb"):
        prep = model.named_steps["prep"]
        clf = model.named_steps["model"]

        Xtr = prep.fit_transform(X_tr)
        Xva = prep.transform(X_va)

        pos = float(y_tr.sum())
        neg = float(len(y_tr) - y_tr.sum())
        clf.set_params(scale_pos_weight=neg / max(pos, 1.0))

        clf.fit(
            Xtr, y_tr,
            eval_set=[(Xva, y_va)],
            verbose=False,
        )
        return clf.predict_proba(Xva)[:, 1], (prep, clf)

    # sklearn models
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1], model


def backtest_cv(builder, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        model = builder(X_tr)
        y_prob, _ = fit_predict(model, X_tr, y_tr, X_va, y_va)

        m = compute_metrics(y_va.to_numpy(), y_prob)
        m["fold"] = fold
        folds.append(m)

    summary = {k: float(np.mean([f[k] for f in folds])) for k in ["roc_auc", "pr_auc", "brier", "ks"]}
    summary |= {k + "_std": float(np.std([f[k] for f in folds])) for k in ["roc_auc", "pr_auc", "brier", "ks"]}
    return folds, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", choices=["rf", "hgb", "xgb"], default="rf")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", type=int, default=5)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    X, y, _ = load_train(args.train)

    builder = {"rf": build_random_forest, "hgb": build_hist_gb, "xgb": build_xgboost}[args.model]

    # 1) CV backtest
    folds, summary = backtest_cv(builder, X, y, n_splits=args.splits)
    save_json({"folds": folds, "summary": summary}, out / "cv_backtest.json")

    # 2) Holdout sanity (also used to pick/inspect plots)
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = builder(X_tr)
    y_prob, fitted_obj = fit_predict(model, X_tr, y_tr, X_va, y_va)

    holdout = compute_metrics(y_va.to_numpy(), y_prob)
    # report both default 0.5 threshold metrics and KS-optimal threshold metrics
    cls_05 = classification_metrics(y_va.to_numpy(), y_prob, threshold=0.5)
    t_ks = best_threshold_by_ks(y_va.to_numpy(), y_prob)
    cls_ks = classification_metrics(y_va.to_numpy(), y_prob, threshold=t_ks)

    save_json({"prob_metrics": holdout, "cls_0.5": cls_05, "cls_ks": cls_ks}, out / "holdout_metrics.json")
    print("Holdout prob metrics:", holdout)
    print("Holdout @0.5:", cls_05)
    print("Holdout @KS*:", cls_ks)

    # Save model
    dump(fitted_obj, out / "best_model.joblib")

    # Curves
    ensure_dir("reports/figures")
    plot_curves(y_va.to_numpy(), y_prob, "reports/figures")
    plot_lift(y_va.to_numpy(), y_prob, "reports/figures")

    print("CV summary:", summary)
    print("Holdout:", holdout)
    print(f"Saved model to {out/'best_model.joblib'}")


if __name__ == "__main__":
    main()