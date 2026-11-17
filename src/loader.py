import pandas as pd


def load_data(path: str = "data/creditcard.csv") -> pd.DataFrame:
    """Load and validate the credit card fraud dataset."""
    df = pd.read_csv(path)

    required = {"Time", "Amount", "Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna()

    n = len(df)
    n_fraud = int(df["Class"].sum())
    pct = df["Class"].mean() * 100
    print(f"Loaded {n:,} transactions | "
          f"Frauds: {n_fraud:,} ({pct:.3f}%)")

    return df
