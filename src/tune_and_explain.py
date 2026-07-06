"""
Day 4 script: Hyperparameter tuning (on Random Forest, our best model so
far) + SHAP explainability.

Run: python src/tune_and_explain.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

DATA_PATH = "data/startups_clean.csv"
MODEL_OUTPUT_DIR = "models"
REPORTS_DIR = "reports"


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in ["state_code"] if c in df.columns])
    X = df.drop(columns=["labels"])
    y = df["labels"]
    return X, y


def tune_random_forest(X_train, y_train):
    # Define a range of settings to try -- RandomizedSearchCV will sample
    # combinations from this instead of trying every single one (faster
    # than GridSearchCV, and usually finds a near-optimal setting anyway).
    param_grid = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_grid,
        n_iter=20,          # try 20 random combinations
        scoring="roc_auc",
        cv=5,                # 5-fold cross-validation for reliable scoring
        random_state=42,
        n_jobs=-1,           # use all CPU cores to speed this up
    )
    search.fit(X_train, y_train)

    print("Best hyperparameters found:")
    print(search.best_params_)
    print(f"Best cross-validation ROC-AUC: {search.best_score_:.3f}")

    return search.best_estimator_


def explain_with_shap(model, X_train, X_test):
    print("\nComputing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, shap_values can be a list [class0, class1]
    # or a single array depending on the sklearn/shap version -- handle both.
    if isinstance(shap_values, list):
        shap_values_for_success = shap_values[1]
    else:
        shap_values_for_success = shap_values

    # Summary plot: shows which features matter most overall, and whether
    # high/low values push the prediction toward success or failure
    plt.figure()
    shap.summary_plot(shap_values_for_success, X_test, show=False)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_summary.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {REPORTS_DIR}/shap_summary.png")

    # Bar chart: simpler view, just overall feature importance
    plt.figure()
    shap.summary_plot(shap_values_for_success, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/shap_importance_bar.png", bbox_inches="tight")
    plt.close()
    print(f"Saved: {REPORTS_DIR}/shap_importance_bar.png")

    return explainer


if __name__ == "__main__":
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tuned_model = tune_random_forest(X_train, y_train)

    predictions_proba = tuned_model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, predictions_proba)
    print(f"\nTuned model ROC-AUC on held-out test set: {test_roc_auc:.3f}")

    explainer = explain_with_shap(tuned_model, X_train, X_test)

    # Save the tuned model (overwrites the earlier best_model.pkl if this
    # one is at least as good)
    joblib.dump(tuned_model, f"{MODEL_OUTPUT_DIR}/best_model.pkl")
    joblib.dump(list(X.columns), f"{MODEL_OUTPUT_DIR}/feature_columns.pkl")
    joblib.dump(explainer, f"{MODEL_OUTPUT_DIR}/shap_explainer.pkl")
    print(f"\nSaved tuned model and SHAP explainer to {MODEL_OUTPUT_DIR}/")