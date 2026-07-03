# A Two-Stage Hybrid Credit Scoring Model: ANN + MARS

Replication of **Lee & Chen (2005)**, *"A two-stage hybrid credit scoring model using
artificial neural networks and multivariate adaptive regression splines,"* Expert Systems
with Applications 28: 743–752 — applied to a real credit-card default dataset
(~45k training rows, ~11k test rows).

## Overview

The paper proposes a two-stage hybrid model: **MARS** is first used to identify the most
significant predictor variables, which are then fed as a reduced input layer into a
**Backpropagation Neural Network (BPN)**. This notebook replicates the full pipeline —
cleaning, feature engineering, EDA, five baseline/benchmark models, the hybrid model,
and paper-consistent evaluation metrics — end to end on this dataset.

## Models Implemented

| # | Model | Role |
|---|---|---|
| 1 | Linear Discriminant Analysis (LDA) | Statistical baseline (Fisher, 1936) |
| 2 | Logistic Regression | Statistical baseline, no normality assumption |
| 3 | MARS | Implemented from scratch (no compiled `py-earth` in this environment) |
| 4 | Backpropagation Neural Network (BPN) | Single hidden layer, trial-and-error node search |
| 5 | **Two-Stage Hybrid (MARS → BPN)** | Paper's core proposal — MARS-selected features feed the BPN |

## Methodology

- **5-fold stratified cross-validation**, matching the paper's evaluation protocol exactly
- **Feature engineering**: 18-variable set analogous to the paper's Table 1 (demographics,
  loan/credit ratios, prior default history)
- **Evaluation**: paper's Table 6 classification comparison (accuracy, Type I/II error) and
  Table 7 **Expected Misclassification Cost** (business-cost-weighted error)
- **Bonus**: the fitted hybrid model is used to score the unlabeled `test.csv` set

## Repository Structure

```
credit-risk-modeling/
├── Credit_Risk_Modeling.ipynb
├── README.md
├── requirements.txt
├── datasets/
│   ├── train.csv
│   └── test.csv
└── images/
    └── (key EDA / results plots)
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook Credit_Risk_Modeling.ipynb
```

Ensure `train.csv` and `test.csv` are present in a `datasets/` folder relative to the
notebook (update the read path at the top of the notebook if your folder structure differs).

## Data

`train.csv` (~45,500 rows) and `test.csv` (~11,400 rows) contain customer-level credit
attributes — demographics, income, employment, existing credit limits/utilization, prior
default history — with `credit_card_default` as the binary target in `train.csv`.

## Key Result

See Sections 11–12 in the notebook for the full model comparison table and expected
misclassification cost. The core question replicated from the paper: does the two-stage
hybrid model outperform standalone MARS, BPN, and the linear statistical baselines —
particularly at catching minority-class (default) cases?

## Author

Yuvraj — Quantitative Options Analyst, Futures First. Built as part of an applied
quant/credit-risk project series ahead of MFE program applications.
