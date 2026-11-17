from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


def train_and_score(df: pd.DataFrame):
    features = [c for c in df.columns if c not in ("Class", "Time")]
    X = df[features]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("\n--- ML holdout metrics (test split) ---")
    print(classification_report(y_test, y_pred, digits=4))

    scores = pd.Series(model.predict_proba(X)[:, 1], index=df.index)
    return scores, report, model
