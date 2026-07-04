"""
Day 2 (extra) script: Merge YC company metadata into the Kaggle dataset.

Matches companies by name (case-insensitive). Most Kaggle startups won't have
a YC match -- that's expected and fine. Where a match exists, we add YC's
batch, industry, and team size as extra features.

Run: python src/merge_yc_data.py
"""

import json
import pandas as pd

RAW_KAGGLE_PATH = "data/startups_raw.csv"   # must still have the "name" column
YC_JSON_PATH = "data/yc_companies.json"
OUTPUT_PATH = "data/startups_with_yc.csv"


def load_kaggle_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} Kaggle startups")
    return df


def load_yc_data(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        yc_raw = json.load(f)
    yc_df = pd.DataFrame(yc_raw)
    print(f"Loaded {len(yc_df)} YC companies")
    # Keep only columns we actually need
    keep_cols = ["name", "batch", "status", "team_size", "industry"]
    keep_cols = [c for c in keep_cols if c in yc_df.columns]
    yc_df = yc_df[keep_cols]
    # Rename to avoid clashing with Kaggle's own "status" column
    yc_df = yc_df.rename(columns={
        "batch": "yc_batch",
        "status": "yc_status",
        "team_size": "yc_team_size",
        "industry": "yc_industry",
    })
    return yc_df


def merge_datasets(kaggle_df: pd.DataFrame, yc_df: pd.DataFrame) -> pd.DataFrame:
    # Normalize names on both sides for matching: lowercase, strip whitespace
    kaggle_df["_match_key"] = kaggle_df["name"].str.lower().str.strip()
    yc_df["_match_key"] = yc_df["name"].str.lower().str.strip()
    yc_df = yc_df.drop(columns=["name"])

    merged = kaggle_df.merge(yc_df, on="_match_key", how="left")
    merged = merged.drop(columns=["_match_key"])

    matched_count = merged["yc_batch"].notna().sum()
    print(f"Matched {matched_count} out of {len(merged)} Kaggle startups to a YC company")
    return merged


if __name__ == "__main__":
    kaggle_df = load_kaggle_data(RAW_KAGGLE_PATH)
    yc_df = load_yc_data(YC_JSON_PATH)
    merged = merge_datasets(kaggle_df, yc_df)

    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved merged data to {OUTPUT_PATH}")
    print(f"Final shape: {merged.shape[0]} rows, {merged.shape[1]} columns")