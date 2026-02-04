from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

ID_COLS = ["customer_id", "name"]
TARGET_COL = "credit_card_default"


def load_train(path: str | Path) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(path)
    ids = df["customer_id"].copy()
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL] + [c for c in ID_COLS if c in df.columns])
    return X, y, ids


def load_test(path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    ids = df["customer_id"].copy()
    X = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    return X, ids


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true = y_true[order]
    pos = (y_true == 1).astype(float)
    neg = (y_true == 0).astype(float)
    pos_cum = np.cumsum(pos) / max(pos.sum(), 1.0)
    neg_cum = np.cumsum(neg) / max(neg.sum(), 1.0)
    return float(np.max(np.abs(pos_cum - neg_cum)))