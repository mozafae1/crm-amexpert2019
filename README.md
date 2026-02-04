# Credit Risk Modeling (AmExpert 2021 CODELAB) — Fast, Reproducible Project

A clean, modular baseline for **credit-card default prediction** using the **AmExpert 2021 CODELAB** dataset.
This repo focuses on **speed + correctness**: strong tree models, disciplined cross-validation backtesting, and standard credit-risk diagnostics.

---

## What’s included

### Modeling
- **Random Forest** baseline (`--model rf`)
- **Gradient Boosting**
  - **XGBoost** (`--model xgb`) if installed (recommended)
  - Fallback: **sklearn HistGradientBoosting** (`--model hgb`)

### Evaluation / Backtesting

- **Threshold-free metrics (primary):**
  - ROC-AUC
  - PR-AUC
  - KS (Kolmogorov–Smirnov)
  - Brier score (calibration)

- Reported via:
  - Stratified K-Fold cross-validation (mean ± std)
  - Holdout (80/20) sanity split

- **Threshold-based metrics (secondary):**
  - Accuracy
  - Balanced Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

- Thresholds:
  - Default 0.5
  - KS-optimal threshold

- **Diagnostic plots:**
  - ROC curve
  - Precision–Recall curve
  - Calibration curve
  - Lift by deciles


### EDA / Visualization
- Target distribution
- Missingness per feature
- Numeric histograms
- Correlation heatmap (numeric)
- Categorical default rates

### Prediction
- Scores `test.csv` (no labels) and writes `submission.csv`:
  - `customer_id`
  - `credit_card_default_prob`

---

## Folder structure

