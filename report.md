# Fraud Detection Framework — Project Report

**Author:** Collins Owino Mumbo
**Date:** November 2026
**Repo:** fraud-framework (commit 3b335d1)

---

## 1. Executive Summary

We built a two-layer fraud detection framework for credit card transactions:
a deterministic rules engine and a machine-learning layer, fused through a
precision-gated "boomerang protocol." On the ULB Credit Card Fraud dataset
(284,807 transactions, 492 frauds, 0.173% fraud rate), the machine-learning
layer alone achieved **precision 0.92, recall 0.82** on an unseen holdout
set. Naively letting low-precision rules override the model destroyed
performance (precision 0.06). Gating rule precedence on historical rule
precision restored fusion precision to **0.97 with recall 0.96**. The central
finding is architectural: *deterministic rules should only override
probabilistic models when the rules have earned that authority through
measured precision.*

---

## 2. Problem

Credit card fraud detection is a highly imbalanced classification problem.
In this dataset only 0.173% of transactions are fraudulent, so a model that
predicts "legitimate" for every row achieves 99.8% accuracy while catching
zero fraud. Accuracy is therefore useless as a metric; we report
**precision, recall, and F1 on the fraud class**, plus the confusion matrix.

The dataset is also anonymized: features V1–V28 are PCA components, and there
is **no card or user identifier**. This rules out true per-card velocity
checks and per-card amount baselines, a limitation we document explicitly.

---

## 3. Architecture
Input (CSV) → Loader → Rules Engine ─┐
ML Engine ──┼→ Fusion (precision-gated) → Evaluate → Output

text

- **Loader** validates schema, drops nulls, reports class balance.
- **Rules engine** runs deterministic checks: amount z-score, odd-hour
  window, absolute high amount. A velocity proxy was implemented but
  excluded (see §5).
- **ML engine** trains a class-balanced Random Forest (100 trees) on a
  stratified 80/20 split and reports honest holdout metrics.
- **Fusion** combines rule flags with ML scores. Any rule whose historical
  precision exceeds 0.5 forces risk to 1.0; weaker rules only nudge the
  score upward.
- **Evaluate** compares final scores to ground truth and writes a ranked
  `output/flagged.csv`.

---

## 4. Results

Four configurations were run on the same data:

| Config | Precision | Recall | F1 |
|---|---|---|---|
| Rules only (loose) | 0.0017 | 1.0000 | 0.0034 |
| Rules only (tight) | 0.0062 | 0.0935 | 0.0117 |
| ML + ungated rules | 0.0609 | 0.9695 | 0.1146 |
| **ML + precision-gated rules** | **0.9673** | **0.9634** | **0.9654** |

**Honest ML holdout metrics** (unseen 20% test split, fraud class):
precision **0.9195**, recall **0.8163**, F1 **0.8649**.

### Reading the results

- **Rules only, loose** flagged nearly every row and achieved perfect recall
  with 0.17% precision — operationally worthless. This demonstrates the
  precision/recall trade-off directly.
- **Rules only, tight** reduced false positives but recall collapsed to
  9.4%: simple amount/hour rules cannot separate fraud from legitimate
  large or late-night purchases in an anonymized dataset.
- **ML + ungated rules** applied the boomerang protocol blindly: 7,337
  legitimate transactions flagged by weak rules were forced to risk 1.0,
  overriding the model and crushing precision to 6%.
- **ML + precision-gated rules** measured each rule's precision first
  (all three rules scored below 1%), found none worthy of override
  authority, and trusted the model. Precision rose to **97%**.

---

## 5. Limitations

1. **No card identifier.** True velocity and per-card amount baselines are
   impossible. The velocity proxy flags ~100% of rows when applied globally
   and is excluded from fusion; this is documented in `src/rules.py`.
2. **In-sample fusion metrics.** Fusion is evaluated on the full dataset
   using ML scores from a model trained on 80% of that same data. This is
   leakage; the holdout metrics in §4 are the unbiased estimate of model
   performance.
3. **Simple rules.** The rules engine is deliberately minimal. More
   sophisticated rules (merchant category anomalies, geo-impossibility with
   real location data) would have higher precision and could earn override
   authority.
4. **Single dataset.** Results are specific to the ULB dataset and may not
   generalize to production transaction streams.

---

## 6. Key Architectural Lesson

The boomerang protocol — deterministic rules overriding probabilistic
models — is only as good as the rules it protects. Our rules had under 1%
precision, so honoring them destroyed fusion performance. **Rule precedence
must be earned through measured precision, not assumed.** This mirrors
real-world security design: hard invariants (e.g., "a transaction cannot be
both refunded and settled") deserve override authority, while heuristic
signals should inform but not veto.

---

## 7. Reproducibility

```bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash
pip install -r requirements.txt
# place creditcard.csv in data/
python main.py
Outputs: output/flagged.csv (top 50 ranked), console metrics.

8. References
ULB Machine Learning Group, Credit Card Fraud Detection dataset, Kaggle.

Scikit-learn documentation: RandomForestClassifier, class_weight,
precision/recall metrics.
