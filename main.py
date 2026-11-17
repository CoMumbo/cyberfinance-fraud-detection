from src.loader import load_data
from src.rules import run_all_rules
from src.fusion import combine, rule_precisions
from src.evaluate import evaluate

USE_ML = True


def main():
    print("=== Fraud Detection Framework ===")
    df = load_data()

    print("\n--- Running rules ---")
    rule_flags = run_all_rules(df)
    for col in rule_flags.columns:
        print(f"  {col}: {int(rule_flags[col].sum()):,} flags")

    ml_scores = None
    if USE_ML:
        from src.ml_engine import train_and_score
        print("\n--- Training ML model ---")
        ml_scores, report, _ = train_and_score(df)

    print("\n--- Rule precision ---")
    weights = rule_precisions(rule_flags, df["Class"])
    for k, v in weights.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Evaluating ---")
    final = combine(rule_flags, ml_scores, rule_weights=weights)
    evaluate(df, final)

    out = df.copy()
    out["risk_score"] = final
    out["reasons"] = rule_flags.apply(
        lambda r: ",".join(r.index[r]), axis=1
    )
    out = out.sort_values("risk_score", ascending=False)
    out.head(50).to_csv("output/flagged.csv", index=False)
    print("\nWrote output/flagged.csv (top 50)")


if __name__ == "__main__":
    main()
