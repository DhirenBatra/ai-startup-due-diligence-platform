"""
Day 3 script: Feature engineering + sector-specific risk logic.

Creates new, more informative columns from existing raw columns, and adds
industry-specific risk flags. Also writes a data dictionary documenting
every column in the final feature set.

Run: python src/feature_engineering.py
"""

import pandas as pd

INPUT_PATH = "data/startups_clean.csv"
OUTPUT_PATH = "data/startups_features.csv"
DICT_PATH = "reports/data_dictionary.md"

df = pd.read_csv(INPUT_PATH)


def add_growth_and_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    # Add a tiny epsilon to avoid division by zero when first/last year are equal
    EPSILON = 0.1

    funding_span = (df["age_last_funding_year"] - df["age_first_funding_year"]).clip(lower=0) + EPSILON
    df["funding_velocity"] = df["funding_total_usd"] / funding_span

    df["funding_per_round"] = df["funding_total_usd"] / df["funding_rounds"].replace(0, 1)

    milestone_span = (df["age_last_milestone_year"] - df["age_first_milestone_year"]).clip(lower=0) + EPSILON
    df["milestone_rate"] = df["milestones"] / milestone_span

    print("Added: funding_velocity, funding_per_round, milestone_rate")
    return df


def add_sector_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    # Biotech: long R&D cycles, regulatory approval risk (e.g. FDA-style approval)
    df["regulatory_risk_flag"] = df.get("is_biotech", 0)

    # Consumer-facing sectors (ecommerce, web) -- more exposed to customer
    # churn and shifting consumer trends than B2B/enterprise sectors
    df["churn_risk_flag"] = (
        df.get("is_ecommerce", 0).fillna(0) + df.get("is_web", 0).fillna(0)
    ).clip(upper=1)

    print("Added: regulatory_risk_flag, churn_risk_flag")
    return df


def write_data_dictionary(df: pd.DataFrame) -> None:
    with open(DICT_PATH, "w") as f:
        f.write("# Data Dictionary — Final Feature Set\n\n")
        f.write(f"Total columns: {len(df.columns)}\n\n")
        f.write("| Column | Type | Description |\n")
        f.write("|---|---|---|\n")

        descriptions = {
            "labels": "Target variable: 1 = acquired/success, 0 = closed/failure",
            "funding_total_usd": "Total funding raised (USD)",
            "funding_rounds": "Number of funding rounds",
            "relationships": "Number of business relationships/connections",
            "milestones": "Number of milestones achieved",
            "is_top500": "Whether the startup is in a top-500 index",
            "funding_velocity": "Engineered: funding raised per year of funding activity",
            "funding_per_round": "Engineered: average funding amount per round",
            "milestone_rate": "Engineered: milestones achieved per year",
            "regulatory_risk_flag": "Engineered: 1 if in a heavily regulated sector (biotech)",
            "churn_risk_flag": "Engineered: 1 if in a consumer-facing, churn-prone sector",
        }

        for col in df.columns:
            dtype = str(df[col].dtype)
            desc = descriptions.get(col, "Raw feature from source dataset")
            f.write(f"| {col} | {dtype} | {desc} |\n")

    print(f"Saved data dictionary to {DICT_PATH}")


if __name__ == "__main__":
    print(f"Starting with {df.shape[1]} columns")

    df = add_growth_and_velocity_features(df)
    df = add_sector_risk_flags(df)

    print(f"\nFinal shape: {df.shape[0]} rows, {df.shape[1]} columns")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

    write_data_dictionary(df)