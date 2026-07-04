"""
Day 1 script: Load the real Kaggle dataset and run basic exploration.

Target column in this dataset is "labels" (1 = success/acquired, 0 = closed).

Run: python src/load_and_explore.py
"""

import pandas as pd

DATA_PATH = DATA_PATH = "data/startups_raw.csv"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {path}")
    return df


def explore(df: pd.DataFrame) -> None:
    print("\n--- All column names ---")
    print(list(df.columns))

    print("\n--- Column data types ---")
    print(df.dtypes)

    print("\n--- Missing values per column (only columns with missing data) ---")
    missing = df.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False))

    print("\n--- Target class balance (labels) ---")
    print(df["labels"].value_counts())
    print(df["labels"].value_counts(normalize=True).round(2))

    print("\n--- Numeric summary (key columns) ---")
    key_numeric_cols = [
        c for c in ["funding_total_usd", "relationships", "funding_rounds", "milestones"]
        if c in df.columns
    ]
    print(df[key_numeric_cols].describe())

    print("\n--- Sample rows ---")
    print(df.head())


if __name__ == "__main__":
    data = load_data(DATA_PATH)
    explore(data)