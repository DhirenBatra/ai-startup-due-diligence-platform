# ===== PART 1: Imports aur API connection setup =====

import os
import json
import pandas as pd
import joblib
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file se API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY nahi mili! Check karo .env file root mein hai ya nahi.")

# OpenRouter OpenAI-compatible endpoint use karta hai
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

# ===== PART 2: Data, Model, SHAP load karna =====

import shap
import numpy as np

# Trained model, feature columns, SHAP explainer load karo
best_model = joblib.load("models/best_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
shap_explainer = joblib.load("models/shap_explainer.pkl")

# Startup dataset load karo (jisme se hum sample startups uthayenge)
df = pd.read_csv("data/startups_clean.csv")

def get_startup_report_data(row_index):
    """
    Ek startup (row_index se) ka data, ML prediction, aur top SHAP factors nikalta hai.
    """
    # Row nikalo aur sirf feature columns rakho (model ko wahi chahiye)
    row = df.iloc[[row_index]][feature_columns]

    # ML prediction: success probability
    probability = best_model.predict_proba(row)[0][1]  # class 1 = success

    # SHAP values nikalo is specific row ke liye
    shap_values = shap_explainer.shap_values(row)

    # SHAP values ka shape normalize karo — kabhi list milta hai, kabhi 3D array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 = success
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # (samples, features, classes) -> class 1 lo

    # Feature name + SHAP contribution ko pair karo, sort karo by absolute impact
    feature_impacts = list(zip(feature_columns, shap_values[0]))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top_factors = feature_impacts[:5]  # top 5 sabse zyada impact wale

    return {
        "raw_data": row.iloc[0].to_dict(),
        "probability": round(float(probability), 3),
        "top_factors": [(name, round(float(val), 4)) for name, val in top_factors]
    }

# ===== PART 3: Prompt design aur Gemini call =====

def build_prompt(startup_data):
    raw = startup_data["raw_data"]
    probability = startup_data["probability"]
    top_factors = startup_data["top_factors"]

    # Top factors ko readable text mein convert karo
    factors_text = "\n".join(
        [f"- {name}: SHAP impact = {val} ({'increases' if val > 0 else 'decreases'} success likelihood)"
         for name, val in top_factors]
    )

    prompt = f"""
You are a startup due-diligence analyst writing a report for an investor.

STARTUP DATA:
{json.dumps(raw, indent=2)}

ML MODEL PREDICTION:
Success probability: {probability * 100:.1f}%

TOP FACTORS DRIVING THIS PREDICTION (from SHAP explainability):
{factors_text}

Write a due-diligence report with EXACTLY these 4 sections, using these exact headers:

## Risk Summary
(2-3 sentences, overall risk level and why, referencing the probability score)

## Strengths
(3-4 bullet points, based on the positive SHAP factors and data)

## Red Flags
(3-4 bullet points, based on negative SHAP factors and any concerning data points)

## Recommendation
(1-2 sentences, a clear investment stance: Proceed, Proceed with Caution, or Pass)

Keep the tone professional, concise, and data-driven. Do not make up information not present in the data above.
"""
    return prompt


def generate_report(row_index):
    startup_data = get_startup_report_data(row_index)
    prompt = build_prompt(startup_data)

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# ===== PART 4: Test on sample startups =====

if __name__ == "__main__":
    # 5 alag-alag startups test karenge (diverse mix ke liye)
    sample_indices = [0, 10, 50, 100, 200]

    all_reports = []

    for idx in sample_indices:
        print(f"\n{'='*60}")
        print(f"Generating report for startup at row index {idx}...")
        print(f"{'='*60}\n")

        try:
            report_text = generate_report(idx)

            # Agar response bahut chhota/incomplete lage, ek baar retry karo
            if len(report_text.strip()) < 200:
                print("Response bahut chhota tha, retry kar rahe hain...")
                report_text = generate_report(idx)

            print(report_text)
            all_reports.append({"row_index": idx, "report": report_text})
        except Exception as e:
            print(f"Error generating report for index {idx}: {e}")

    # Sab reports ek file mein save kar do
    with open("reports/sample_llm_reports.txt", "w", encoding="utf-8") as f:
        for item in all_reports:
            f.write(f"\n{'='*60}\n")
            f.write(f"STARTUP ROW INDEX: {item['row_index']}\n")
            f.write(f"{'='*60}\n")
            f.write(item["report"])
            f.write("\n")

    print(f"\n\nDone! {len(all_reports)} reports saved to reports/sample_llm_reports.txt")