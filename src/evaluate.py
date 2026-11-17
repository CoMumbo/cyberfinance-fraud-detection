import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate(df: pd.DataFrame, final_scores: pd.Series,
             threshold: float = 0.5) -> dict:
    """Compare final scores against ground truth labels."""
    preds = (final_scores >= threshold).astype(int)
    y = df["Class"]

    cm = confusion_matrix(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print("Confusion matrix:")
    print(cm)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
