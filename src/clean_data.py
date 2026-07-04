"""
Day 2 script: Clean the raw dataset and prepare it for ML models.

Run: python src/clean_data.py
"""

import pandas as pd

RAW_PATH = "data/startups_raw.csv"
CLEAN_PATH = "data/startups_clean.csv"

# Columns that don't help prediction: IDs, raw text, duplicate/leftover columns
COLUMNS_TO_DROP = [
    "Unnamed: 0", "Unnamed: 6", "latitude", "longitude", "zip_code",
    "id", "city", "name", "object_id", "state_code.1",
    "founded_at", "closed_at", "first_funding_at", "last_funding_at",
    "status",  # we keep "labels" (0/1) instead, since it's already numeric
]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def drop_unneeded_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_present = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_present)
    print(f"Dropped {len(cols_present)} columns. {len(df.columns)} columns remain.")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # age_first/last_milestone_year missing likely means the startup had no
    # milestones yet -- fill with 0 rather than guessing a fake age.
    for col in ["age_first_milestone_year", "age_last_milestone_year"]:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            df[col] = df[col].fillna(0)
            print(f"Filled {missing_count} missing values in '{col}' with 0")
    return df


def encode_category_code(df: pd.DataFrame) -> pd.DataFrame:
    # category_code (industry) is text -- convert to numeric "dummy" columns
    # so the ML model can use it. Note: is_software/is_web/etc columns
    # already exist for this purpose, so we drop the raw text version.
    if "category_code" in df.columns:
        df = df.drop(columns=["category_code"])
        print("Dropped 'category_code' (already represented by is_software/is_web/etc columns)")
    return df


if __name__ == "__main__":
    data = load_data(RAW_PATH)
    data = drop_unneeded_columns(data)
    data = fill_missing_values(data)
    data = encode_category_code(data)

    print(f"\nFinal shape: {data.shape[0]} rows, {data.shape[1]} columns")
    print("\nRemaining missing values:")
    print(data.isnull().sum().sum(), "total missing values left")

    data.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned data to {CLEAN_PATH}")