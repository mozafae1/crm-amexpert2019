# Credit Risk Modeling using Random Forest & XGBoost (AmExpert 2021 CODELAB)

This repository implements a credit risk modeling pipeline for the AmExpert 2021 CODELAB dataset using Random Forest and XGBoost. The project follows a structured approach, covering exploratory data analysis, model training, cross-validated backtesting, and final scoring on an unseen test set. Model performance is evaluated using metrics and diagnostics that are standard in credit risk analysis, including ROC-AUC, PR-AUC, KS, calibration measures, and lift analysis.

---

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

## Project structure
```text
crm_amexpert2019/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── models/
├── reports/
│   └── figures/
├── src/
│   └── credit_risk/
│       ├── __init__.py
│       ├── utils.py
│       ├── preprocess.py
│       ├── eda.py
│       ├── models.py
│       ├── evaluate.py
│       ├── train.py
│       └── predict.py
├── requirements.txt
└── README.md
