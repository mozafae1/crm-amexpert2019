from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from .utils import ensure_dir, ks_statistic


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ks": float(ks_statistic(y_true.astype(int), y_prob)),
    }


def plot_curves(y_true: np.ndarray, y_prob: np.ndarray, outdir: str):
    outdir = ensure_dir(outdir)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC curve")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    fig.savefig(outdir / "roc_curve.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    p, r, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(r, p)
    plt.title("Precision–Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    fig.savefig(outdir / "pr_curve.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    # Calibration curve (optional)
    try:
        from sklearn.calibration import calibration_curve

        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="quantile"
        )
        fig = plt.figure()
        plt.plot(mean_pred, frac_pos, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title("Calibration curve")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        fig.savefig(outdir / "calibration_curve.png", bbox_inches="tight", dpi=160)
        plt.close(fig)
    except Exception:
        pass


def plot_lift(y_true: np.ndarray, y_prob: np.ndarray, outdir: str, bins: int = 10):
    outdir = ensure_dir(outdir)

    order = np.argsort(-y_prob)
    y_true = y_true[order]
    n = len(y_true)
    edges = np.linspace(0, n, bins + 1, dtype=int)

    overall = y_true.mean()
    lift = []
    for i in range(bins):
        s, e = edges[i], edges[i + 1]
        lift.append(y_true[s:e].mean() / max(overall, 1e-9))

    fig = plt.figure()
    plt.plot(range(1, bins + 1), lift, marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.title("Lift by score decile (1=highest risk)")
    plt.xlabel("Decile")
    plt.ylabel("Lift")
    fig.savefig(outdir / "lift_deciles.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_hat = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "f1": float(f1_score(y_true, y_hat, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    
def best_threshold_by_ks(y_true: np.ndarray, y_prob: np.ndarray, grid_size: int = 200) -> float:
    thresholds = np.linspace(0.0, 1.0, grid_size)
    best_t, best_ks = 0.5, -1.0
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        # convert predictions to score ordering proxy: use prob directly for KS already computed,
        # but for threshold selection we approximate via TPR-FPR (same as KS on ROC)
        # KS = max(TPR - FPR)
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ks_values = tpr - fpr
    idx = int(np.argmax(ks_values))
    return float(thr[idx])