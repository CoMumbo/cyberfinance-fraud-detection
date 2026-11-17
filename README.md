# Cyberfinance Fraud Detection Framework

A two-layer fraud detection system for credit card transactions: a
deterministic rules engine and a machine-learning layer, fused through a
precision-gated "boomerang protocol."

## Results

| Config | Precision | Recall | F1 |
|---|---|---|---|
| Rules only (loose) | 0.0017 | 1.0000 | 0.0034 |
| Rules only (tight) | 0.0062 | 0.0935 | 0.0117 |
| ML + ungated rules | 0.0609 | 0.9695 | 0.1146 |
| **ML + precision-gated rules** | **0.9673** | **0.9634** | **0.9654** |

Honest ML holdout metrics on the unseen 20% test split (fraud class):
**precision 0.9195, recall 0.8163, F1 0.8649**.

## Architecture
Input (CSV) → Loader → Rules Engine ─┐
ML Engine ──┼→ Fusion (precision-gated) → Evaluate → Output

text

- **Loader** — validates schema, drops nulls, reports class balance.
- **Rules engine** — amount z-score, odd-hour window, absolute high amount.
- **ML engine** — class-balanced Random Forest on a stratified 80/20 split.
- **Fusion** — a rule only forces risk=1.0 if its measured precision
  exceeds 0.5; weaker rules nudge the score upward but do not override ML.
- **Evaluate** — precision, recall, F1, confusion matrix, ranked output.

## Key finding

The boomerang protocol (rules override ML) is only as good as the rules it
protects. Our rules had under 1% precision, so honoring them blindly dropped
fusion precision to 6%. Precision-gating rule authority restored it to 97%.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt
# Download creditcard.csv from Kaggle (mlg-ulb/creditcardfraud)
# and place it in data/
python main.py
Dataset
ULB Machine Learning Group, Credit Card Fraud Detection:
284,807 transactions, 492 frauds (0.173%).
Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Limitations
No card identifier in the dataset; true per-card velocity is impossible.

Fusion metrics use in-sample ML scores (leakage); holdout metrics are the
unbiased estimate.

Rules are deliberately minimal; richer rules would earn override authority.

Full write-up: report.md.
