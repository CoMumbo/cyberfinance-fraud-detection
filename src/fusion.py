import pandas as pd


def rule_precisions(rule_flags: pd.DataFrame, y: pd.Series) -> dict:
    """Compute each rule's precision against ground truth."""
    out = {}
    for col in rule_flags.columns:
        flagged = rule_flags[col]
        if flagged.sum() == 0:
            out[col] = 0.0
        else:
            out[col] = float(y[flagged].mean())
    return out


def combine(rule_flags: pd.DataFrame,
            ml_scores=None,
            rule_weights: dict | None = None,
            min_rule_precision: float = 0.5) -> pd.Series:
    """
    Weighted fusion with precision-gated boomerang protocol.

    A rule only forces risk=1.0 if its historical precision exceeds
    min_rule_precision. Weak rules contribute partial risk only.
    """
    if ml_scores is not None:
        final = ml_scores.copy()
    else:
        final = pd.Series(0.0, index=rule_flags.index)

    if rule_weights is None:
        rule_weights = {c: 1.0 for c in rule_flags.columns}

    for col in rule_flags.columns:
        flagged = rule_flags[col]
        prec = rule_weights.get(col, 0.0)
        if prec >= min_rule_precision:
            final[flagged] = 1.0
        else:
            final[flagged] = (final[flagged] + 0.5 * prec).clip(upper=0.99)

    return final
