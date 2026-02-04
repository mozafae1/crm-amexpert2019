from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from .preprocess import make_preprocessor


def build_random_forest(X) -> Pipeline:
    prep = make_preprocessor(X)
    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
    )
    return Pipeline([("prep", prep), ("model", clf)])


def build_hist_gb(X) -> Pipeline:
    prep = make_preprocessor(X)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=6,
        max_iter=600,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline([("prep", prep), ("model", clf)])


def build_xgboost(X) -> Pipeline:
    prep = make_preprocessor(X)
    from xgboost import XGBClassifier

    clf = XGBClassifier(
    n_estimators=4000,          # will be cut by early stopping
    learning_rate=0.03,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=1.0,
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,

    early_stopping_rounds=150,  # <-- MOVE IT HERE
)
    return Pipeline([("prep", prep), ("model", clf)])