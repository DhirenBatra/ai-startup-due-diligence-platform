"""
Day 3 script: Train and compare three ML models --
Logistic Regression, Random Forest, and XGBoost.

Logistic Regression gets scaled features (it's sensitive to feature scale);
Random Forest and XGBoost use raw features (tree-based models don't need
scaling since they split on thresholds, not distances).

Run: python src/train_model.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import joblib

DATA_PATH = "data/startups_clean.csv"
MODEL_OUTPUT_DIR = "models"


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)

    non_feature_cols = ["state_code"]
    df = df.drop(columns=[c for c in non_feature_cols if c in df.columns])

    X = df.drop(columns=["labels"])
    y = df["labels"]

    print(f"Features: {X.shape[1]} columns")
    print(f"Target: 'labels' ({y.sum()} success, {len(y) - y.sum()} failure)")

    return X, y


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"ROC-AUC:  {roc_auc:.3f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}")

    return {"name": name, "model": model, "accuracy": accuracy, "f1": f1, "roc_auc": roc_auc}


if __name__ == "__main__":
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {len(X_train)} rows | Test set: {len(X_test)} rows")

    # Scale features for Logistic Regression only -- it's distance/weight
    # based, so a column like funding_total_usd (in millions) would otherwise
    # dominate a column like has_VC (0 or 1). Tree models don't need this.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    results.append(train_and_evaluate(
        "Logistic Regression (scaled)",
        LogisticRegression(max_iter=1000),
        X_train_scaled, X_test_scaled, y_train, y_test
    ))

    results.append(train_and_evaluate(
        "Random Forest",
        RandomForestClassifier(n_estimators=200, random_state=42),
        X_train, X_test, y_train, y_test
    ))

    results.append(train_and_evaluate(
        "XGBoost",
        XGBClassifier(n_estimators=200, random_state=42, eval_metric="logloss"),
        X_train, X_test, y_train, y_test
    ))

    print("\n\n=== MODEL COMPARISON ===")
    comparison_df = pd.DataFrame([
        {"Model": r["name"], "Accuracy": r["accuracy"], "F1": r["f1"], "ROC-AUC": r["roc_auc"]}
        for r in results
    ])
    print(comparison_df.to_string(index=False))

    best = max(results, key=lambda r: r["roc_auc"])
    joblib.dump(best["model"], f"{MODEL_OUTPUT_DIR}/best_model.pkl")
    joblib.dump(list(X.columns), f"{MODEL_OUTPUT_DIR}/feature_columns.pkl")
    joblib.dump(scaler, f"{MODEL_OUTPUT_DIR}/scaler.pkl")
    print(f"\nBest model: {best['name']} (ROC-AUC: {best['roc_auc']:.3f}) -- saved to models/best_model.pkl")