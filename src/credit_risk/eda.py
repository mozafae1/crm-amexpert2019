from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import pandas as pd

from .utils import ensure_dir, load_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = ensure_dir(args.out)
    X, y, _ = load_train(args.train)

    # Target distribution
    fig = plt.figure()
    y.value_counts().sort_index().plot(kind="bar")
    plt.title("Target distribution (credit_card_default)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    fig.savefig(out / "target_distribution.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    # Missingness
    miss = X.isna().mean().sort_values(ascending=False)
    fig = plt.figure(figsize=(7, 4))
    miss[miss > 0].plot(kind="bar")
    plt.title("Missingness rate per feature")
    plt.ylabel("Missing fraction")
    fig.savefig(out / "missingness.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    # Numeric histograms
    num_cols = [c for c in X.columns if X[c].dtype != "object"]
    for c in num_cols:
        fig = plt.figure()
        X[c].hist(bins=40)
        plt.title(f"Histogram: {c}")
        fig.savefig(out / f"hist_{c}.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

    # Correlation heatmap
    num = X.select_dtypes(exclude=["object"])
    if num.shape[1] >= 2:
        corr = num.corr(numeric_only=True)
        fig = plt.figure(figsize=(8, 6))
        plt.imshow(corr, aspect="auto")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation heatmap (numeric features)")
        plt.colorbar()
        fig.savefig(out / "corr_heatmap.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

    # Categorical default rates
    df = X.copy()
    df["target"] = y.values
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    for c in cat_cols:
        rate = df.groupby(c)["target"].mean().sort_values(ascending=False)
        fig = plt.figure(figsize=(7, 4))
        rate.plot(kind="bar")
        plt.title(f"Default rate by {c}")
        plt.ylabel("Default rate")
        fig.savefig(out / f"default_rate_{c}.png", bbox_inches="tight", dpi=160)
        plt.close(fig)

    print(f"Saved figures to {out}")


if __name__ == "__main__":
    main()