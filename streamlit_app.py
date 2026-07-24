# streamlit_app.py: Frontend dashboard for the AI Startup Due Diligence Platform

import streamlit as st
import requests
import pandas as pd

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Startup Due Diligence Platform", layout="wide")

st.title("AI Startup Due Diligence Platform")
st.write("Upload a pitch deck or enter startup metrics to get an ML-driven success prediction and an AI-generated due diligence report.")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict Startup", "Upload Pitch Deck", "History"])

if page == "Predict Startup":
    st.header("Predict Startup Success")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age_first_funding_year = st.number_input("Age at first funding (years)", value=1.0)
            age_last_funding_year = st.number_input("Age at last funding (years)", value=2.0)
            age_first_milestone_year = st.number_input("Age at first milestone (years)", value=1.5)
            age_last_milestone_year = st.number_input("Age at last milestone (years)", value=3.0)
            relationships = st.number_input("Relationships", value=5, step=1)
            funding_rounds = st.number_input("Funding rounds", value=2, step=1)
            funding_total_usd = st.number_input("Total funding (USD)", value=1000000, step=100000)
            milestones = st.number_input("Milestones", value=2, step=1)
            avg_participants = st.number_input("Avg participants per round", value=2.0)
            is_top500 = st.checkbox("Top 500 startup")

        with col2:
            st.write("State")
            is_CA = st.checkbox("California")
            is_NY = st.checkbox("New York")
            is_MA = st.checkbox("Massachusetts")
            is_TX = st.checkbox("Texas")
            is_otherstate = st.checkbox("Other state")

            st.write("Category")
            is_software = st.checkbox("Software")
            is_web = st.checkbox("Web")
            is_mobile = st.checkbox("Mobile")
            is_enterprise = st.checkbox("Enterprise")
            is_advertising = st.checkbox("Advertising")

        with col3:
            is_gamesvideo = st.checkbox("Games/Video")
            is_ecommerce = st.checkbox("E-commerce")
            is_biotech = st.checkbox("Biotech")
            is_consulting = st.checkbox("Consulting")
            is_othercategory = st.checkbox("Other category")

            st.write("Funding type")
            has_VC = st.checkbox("Has VC backing")
            has_angel = st.checkbox("Has angel backing")
            has_roundA = st.checkbox("Completed Round A")
            has_roundB = st.checkbox("Completed Round B")
            has_roundC = st.checkbox("Completed Round C")
            has_roundD = st.checkbox("Completed Round D")

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "age_first_funding_year": age_first_funding_year,
            "age_last_funding_year": age_last_funding_year,
            "age_first_milestone_year": age_first_milestone_year,
            "age_last_milestone_year": age_last_milestone_year,
            "relationships": relationships,
            "funding_rounds": funding_rounds,
            "funding_total_usd": funding_total_usd,
            "milestones": milestones,
            "is_CA": int(is_CA),
            "is_NY": int(is_NY),
            "is_MA": int(is_MA),
            "is_TX": int(is_TX),
            "is_otherstate": int(is_otherstate),
            "is_software": int(is_software),
            "is_web": int(is_web),
            "is_mobile": int(is_mobile),
            "is_enterprise": int(is_enterprise),
            "is_advertising": int(is_advertising),
            "is_gamesvideo": int(is_gamesvideo),
            "is_ecommerce": int(is_ecommerce),
            "is_biotech": int(is_biotech),
            "is_consulting": int(is_consulting),
            "is_othercategory": int(is_othercategory),
            "has_VC": int(has_VC),
            "has_angel": int(has_angel),
            "has_roundA": int(has_roundA),
            "has_roundB": int(has_roundB),
            "has_roundC": int(has_roundC),
            "has_roundD": int(has_roundD),
            "avg_participants": avg_participants,
            "is_top500": int(is_top500),
        }

        with st.spinner("Running prediction..."):
            response = requests.post(f"{API_BASE_URL}/predict", json=payload)

        if response.status_code == 200:
            result = response.json()
            st.session_state["last_prediction"] = result
            st.session_state["last_payload"] = payload

            st.success(f"Success Probability: {result['success_probability'] * 100:.1f}%")

            st.subheader("Top Contributing Factors")

            factors_df = pd.DataFrame(result["top_factors"])
            factors_df = factors_df.set_index("feature")
            st.bar_chart(factors_df["shap_impact"])

            for factor in result["top_factors"]:
                direction = "increases" if factor["shap_impact"] > 0 else "decreases"
                st.write(f"**{factor['feature']}**: {factor['shap_impact']:.4f} ({direction} success likelihood)")
        else:
            st.error(f"Prediction failed: {response.text}")

    if "last_prediction" in st.session_state:
        st.divider()
        if st.button("Generate Due Diligence Report"):
            report_payload = {
                "startup_data": st.session_state["last_payload"],
                "success_probability": st.session_state["last_prediction"]["success_probability"],
                "top_factors": st.session_state["last_prediction"]["top_factors"],
            }

            with st.spinner("Generating report... this may take a few seconds"):
                report_response = requests.post(f"{API_BASE_URL}/generate-report", json=report_payload)

            if report_response.status_code == 200:
                report_text = report_response.json()["report"]
                st.subheader("Due Diligence Report")
                st.markdown(report_text)
            else:
                st.error(f"Report generation failed: {report_response.text}")

elif page == "Upload Pitch Deck":
    st.header("Upload Pitch Deck")
    st.write("This section will be built next.")

elif page == "History":
    st.header("Startup History")
    st.write("This section will be built next.")