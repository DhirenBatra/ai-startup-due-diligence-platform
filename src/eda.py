"""
Day 2 script: Exploratory Data Analysis (EDA) -- distributions and
correlations. Saves chart images to the reports/ folder so they can be
included in the final project report.

Run: python src/eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "data/startups_clean.csv"
OUTPUT_DIR = "reports"

df = pd.read_csv(DATA_PATH)


def plot_class_balance():
    plt.figure(figsize=(5, 4))
    sns.countplot(x="labels", data=df)
    plt.title("Class Balance: Success (1) vs Failure (0)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/class_balance.png")
    plt.close()
    print("Saved: reports/class_balance.png")


def plot_key_distributions():
    key_cols = ["funding_total_usd", "relationships", "funding_rounds", "milestones"]
    key_cols = [c for c in key_cols if c in df.columns]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, col in enumerate(key_cols):
        sns.histplot(df[col], bins=30, ax=axes[i], kde=True)
        axes[i].set_title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/key_distributions.png")
    plt.close()
    print("Saved: reports/key_distributions.png")


def plot_correlation_heatmap():
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap (all numeric features)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png")
    plt.close()
    print("Saved: reports/correlation_heatmap.png")


def print_top_correlations_with_target():
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    corr_with_target = numeric_df.corr()["labels"].drop("labels").sort_values(ascending=False)
    print("\n--- Features most correlated with success (labels) ---")
    print(corr_with_target.head(10))
    print("\n--- Features most negatively correlated with success ---")
    print(corr_with_target.tail(5))


if __name__ == "__main__":
    plot_class_balance()
    plot_key_distributions()
    plot_correlation_heatmap()
    print_top_correlations_with_target()