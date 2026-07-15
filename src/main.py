# main.py: FastAPI application entry point

import os
import json
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

app = FastAPI(title="AI Startup Due Diligence Platform")

# Load model artifacts once when the server starts, not on every request
best_model = joblib.load("models/best_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
shap_explainer = joblib.load("models/shap_explainer.pkl")

# LLM client setup (OpenRouter, OpenAI-compatible)
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
llm_client = OpenAI(api_key=openrouter_api_key, base_url="https://openrouter.ai/api/v1")


class StartupFeatures(BaseModel):
    age_first_funding_year: float
    age_last_funding_year: float
    age_first_milestone_year: float
    age_last_milestone_year: float
    relationships: int
    funding_rounds: int
    funding_total_usd: int
    milestones: int
    is_CA: int
    is_NY: int
    is_MA: int
    is_TX: int
    is_otherstate: int
    is_software: int
    is_web: int
    is_mobile: int
    is_enterprise: int
    is_advertising: int
    is_gamesvideo: int
    is_ecommerce: int
    is_biotech: int
    is_consulting: int
    is_othercategory: int
    has_VC: int
    has_angel: int
    has_roundA: int
    has_roundB: int
    has_roundC: int
    has_roundD: int
    avg_participants: float
    is_top500: int


class ReportRequest(BaseModel):
    startup_data: dict
    success_probability: float
    top_factors: list


def build_prompt(startup_data, probability, top_factors):
    factors_text = "\n".join(
        [f"- {f['feature']}: SHAP impact = {f['shap_impact']} "
         f"({'increases' if f['shap_impact'] > 0 else 'decreases'} success likelihood)"
         for f in top_factors]
    )

    prompt = f"""
You are a startup due-diligence analyst writing a report for an investor.

STARTUP DATA:
{json.dumps(startup_data, indent=2)}

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


@app.get("/")
def read_root():
    return {"status": "API is running"}


@app.post("/predict")
def predict(startup: StartupFeatures):
    input_dict = startup.dict()
    input_row = pd.DataFrame([[input_dict[col] for col in feature_columns]], columns=feature_columns)

    probability = best_model.predict_proba(input_row)[0][1]

    shap_values = shap_explainer.shap_values(input_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    feature_impacts = list(zip(feature_columns, shap_values[0]))
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top_factors = feature_impacts[:5]

    return {
        "success_probability": round(float(probability), 3),
        "top_factors": [
            {"feature": name, "shap_impact": round(float(val), 4)}
            for name, val in top_factors
        ]
    }


@app.post("/generate-report")
def generate_report(request: ReportRequest):
    prompt = build_prompt(request.startup_data, request.success_probability, request.top_factors)

    response = llm_client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    report_text = response.choices[0].message.content

    if len(report_text.strip()) < 200:
        response = llm_client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        report_text = response.choices[0].message.content

    return {"report": report_text}