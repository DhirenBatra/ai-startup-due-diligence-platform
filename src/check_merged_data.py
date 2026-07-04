"""
Quick check script: verify the merged dataset looks correct before moving on.

Run: python src/check_merged_data.py
"""

import pandas as pd

MERGED_PATH = "data/startups_with_yc.csv"

df = pd.read_csv(MERGED_PATH)

print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

print("--- New YC columns added ---")
yc_cols = [c for c in df.columns if c.startswith("yc_")]
print(yc_cols)

print("\n--- How many rows actually got YC data? ---")
print(df[yc_cols].notna().sum())

print("\n--- Show only the rows that DID get matched (sample) ---")
matched = df[df["yc_batch"].notna()]
print(matched[["name", "labels", "yc_batch", "yc_industry", "yc_team_size"]].head(10))

print("\n--- Confirm original Kaggle columns are still intact ---")
print("funding_total_usd" in df.columns, "milestones" in df.columns, "labels" in df.columns)