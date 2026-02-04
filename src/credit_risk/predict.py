from __future__ import annotations

import argparse
import pandas as pd
from joblib import load

from .utils import load_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    X_test, ids = load_test(args.test)
    obj = load(args.model_path)

    if isinstance(obj, tuple):
        prep, clf = obj
        Xt = prep.transform(X_test)
        prob = clf.predict_proba(Xt)[:, 1]
    else:
        prob = obj.predict_proba(X_test)[:, 1]

    out = pd.DataFrame(
        {"customer_id": ids, "credit_card_default_prob": prob}
    )
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()