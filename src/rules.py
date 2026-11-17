import pandas as pd


def rule_amount_zscore(df: pd.DataFrame, threshold: float = 5.0):
    """Flag transactions whose amount is > threshold std deviations from mean."""
    z = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    return z > threshold, "amount_zscore"


def rule_odd_hour(df: pd.DataFrame, start: int = 3, end: int = 4):
    """Flag transactions between 03:00 and 04:00."""
    hour = (df["Time"] // 3600) % 24
    return hour.between(start, end), "odd_hour"


def rule_high_amount(df: pd.DataFrame, threshold: float = 2000.0):
    """Flag any transaction above an absolute amount."""
    return df["Amount"] > threshold, "high_amount"


def rule_velocity(df: pd.DataFrame, window_seconds: int = 600,
                  max_tx: int = 50):
    """
    Proxy velocity: flag timestamps where >max_tx transactions occur
    within a window_seconds window.

    NOTE: this dataset has no card/user ID, so true per-card velocity is
    impossible. This is a global proxy and should be documented as a
    limitation in the report. It is intentionally excluded from
    run_all_rules because, applied globally, it flags nearly every row.
    """
    times = df["Time"].values
    sorted_times = sorted(times)

    flagged_times = set()
    n = len(sorted_times)
    j = 0
    for i in range(n):
        while j < n and sorted_times[j] <= sorted_times[i] + window_seconds:
            j += 1
        if (j - i) > max_tx:
            flagged_times.add(sorted_times[i])

    mask = df["Time"].isin(flagged_times)
    return mask, "velocity_proxy"


def run_all_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Run every rule and return a DataFrame of boolean flags.

    velocity_proxy is deliberately excluded: without a card ID, a global
    velocity check flags ~100% of rows and destroys precision.
    """
    results = {}
    for fn in (rule_amount_zscore, rule_odd_hour, rule_high_amount):
        flag, name = fn(df)
        results[name] = flag
    return pd.DataFrame(results)
