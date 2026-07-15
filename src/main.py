# main.py: FastAPI application entry point

import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Startup Due Diligence Platform")

best_model = joblib.load("models/best_model.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")
shap_explainer = joblib.load("models/shap_explainer.pkl")


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