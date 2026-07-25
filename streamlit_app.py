# streamlit_app.py: Frontend dashboard for the AI Startup Due Diligence Platform

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

API_BASE_URL = "http://127.0.0.1:8000"

FEATURE_LABELS = {
    "age_first_funding_year": "Years to First Funding",
    "age_last_funding_year": "Years to Last Funding",
    "age_first_milestone_year": "Years to First Milestone",
    "age_last_milestone_year": "Years to Last Milestone",
    "relationships": "Relationships",
    "funding_rounds": "Funding Rounds",
    "funding_total_usd": "Total Funding (USD)",
    "milestones": "Milestones Achieved",
    "is_CA": "Based in California",
    "is_NY": "Based in New York",
    "is_MA": "Based in Massachusetts",
    "is_TX": "Based in Texas",
    "is_otherstate": "Based in Other State",
    "is_software": "Software Category",
    "is_web": "Web Category",
    "is_mobile": "Mobile Category",
    "is_enterprise": "Enterprise Category",
    "is_advertising": "Advertising Category",
    "is_gamesvideo": "Games/Video Category",
    "is_ecommerce": "E-commerce Category",
    "is_biotech": "Biotech Category",
    "is_consulting": "Consulting Category",
    "is_othercategory": "Other Category",
    "has_VC": "VC Backing",
    "has_angel": "Angel Backing",
    "has_roundA": "Completed Round A",
    "has_roundB": "Completed Round B",
    "has_roundC": "Completed Round C",
    "has_roundD": "Completed Round D",
    "avg_participants": "Avg Participants per Round",
    "is_top500": "Top 500 Startup Status",
}

st.set_page_config(
    page_title="AI Startup Due Diligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #F5F3FF 0%, #E0E7FF 50%, #FCE7F3 100%);
}

[data-testid="stHeader"] {
    background: transparent;
}

h1 {
    background: linear-gradient(90deg, #7C3AED, #DB2777);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

h2, h3 {
    color: #6D28D9;
}

[data-testid="stForm"] {
    background-color: #FFFFFF;
    padding: 25px;
    border-radius: 15px;
    border: 2px solid #C4B5FD;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.15);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #DDD6FE 0%, #FBCFE8 100%);
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #8B5CF6, #DB2777);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
}

[data-testid="stMetric"] label {
    color: #F5F3FF !important;
}

[data-testid="stMetric"] div {
    color: #FFFFFF !important;
}

.stButton button, .stFormSubmitButton button {
    background: linear-gradient(90deg, #8B5CF6, #DB2777);
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: 600;
    padding: 10px 24px;
}

.stButton button:hover, .stFormSubmitButton button:hover {
    background: linear-gradient(90deg, #7C3AED, #BE185D);
    color: white;
    box-shadow: 0 4px 12px rgba(139, 92, 246, 0.4);
}

[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

st.title("AI Startup Due Diligence Platform")
st.write("Upload a pitch deck or enter startup metrics to get an ML-driven success prediction and an AI-generated due diligence report.")

st.sidebar.title("Navigation")
st.sidebar.write("AI-powered startup evaluation using ML predictions, SHAP explainability, and LLM-generated reports.")
st.sidebar.divider()
page = st.sidebar.radio("Go to", ["Predict Startup", "Upload Pitch Deck", "History"])
st.sidebar.divider()
st.sidebar.caption("Built with FastAPI, scikit-learn, XGBoost, SHAP, and OpenRouter LLM integration.")

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

            probability_pct = result['success_probability'] * 100

            if probability_pct >= 70:
                verdict = "Strong"
                gauge_color = "#059669"
            elif probability_pct >= 40:
                verdict = "Moderate"
                gauge_color = "#D97706"
            else:
                verdict = "Weak"
                gauge_color = "#DC2626"

            gauge_col, side_col = st.columns([2, 1])

            with gauge_col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability_pct,
                    number={'suffix': "%", 'font': {'size': 40}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 40], 'color': "#FEE2E2"},
                            {'range': [40, 70], 'color': "#FEF3C7"},
                            {'range': [70, 100], 'color': "#D1FAE5"},
                        ],
                    },
                    title={'text': "Success Probability"}
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with side_col:
                st.metric(label="Verdict", value=verdict)
                top_feature_raw = result["top_factors"][0]["feature"]
                st.metric(label="Top Factor", value=FEATURE_LABELS.get(top_feature_raw, top_feature_raw))

            st.subheader("Top Contributing Factors")

            factors_df = pd.DataFrame(result["top_factors"])
            factors_df["readable_feature"] = factors_df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
            bar_colors = ["#059669" if val > 0 else "#DC2626" for val in factors_df["shap_impact"]]

            fig_bar = go.Figure(go.Bar(
                x=factors_df["readable_feature"],
                y=factors_df["shap_impact"],
                marker_color=bar_colors
            ))
            fig_bar.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis_title="SHAP Impact",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            for factor in result["top_factors"]:
                direction = "increases" if factor["shap_impact"] > 0 else "decreases"
                readable_name = FEATURE_LABELS.get(factor["feature"], factor["feature"])
                st.write(f"**{readable_name}**: {factor['shap_impact']:.4f} ({direction} success likelihood)")
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

                sections = {"Risk Summary": "", "Strengths": "", "Red Flags": "", "Recommendation": ""}
                current_section = None
                for line in report_text.split("\n"):
                    stripped = line.strip().lstrip("#").strip()
                    if stripped in sections:
                        current_section = stripped
                    elif current_section:
                        sections[current_section] += line + "\n"

                st.markdown(f"**Risk Summary**\n\n{sections['Risk Summary']}")

                col_strength, col_flags = st.columns(2)
                with col_strength:
                    st.markdown(
                        f"<div style='background-color:#D1FAE5; padding:15px; border-radius:10px;'>"
                        f"<h4 style='color:#065F46;'>Strengths</h4>{sections['Strengths']}</div>",
                        unsafe_allow_html=True
                    )
                with col_flags:
                    st.markdown(
                        f"<div style='background-color:#FEE2E2; padding:15px; border-radius:10px;'>"
                        f"<h4 style='color:#991B1B;'>Red Flags</h4>{sections['Red Flags']}</div>",
                        unsafe_allow_html=True
                    )

                st.markdown(f"**Recommendation**\n\n{sections['Recommendation']}")
            else:
                st.error(f"Report generation failed: {report_response.text}")

elif page == "Upload Pitch Deck":
    st.header("Upload Pitch Deck")
    st.write("Upload a PDF pitch deck to extract key figures and team information.")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Extract Data"):
            with st.spinner("Extracting text and running OCR if needed... this may take up to a minute"):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                extract_response = requests.post(f"{API_BASE_URL}/upload-pitch-deck", files=files)

            if extract_response.status_code == 200:
                st.session_state["extracted_data"] = extract_response.json()
                st.success("Extraction complete.")
            else:
                st.error(f"Extraction failed: {extract_response.text}")

    if "extracted_data" in st.session_state:
        data = st.session_state["extracted_data"]

        st.subheader("Extracted Data (editable)")
        st.write("Review and correct any values below if the automatic extraction made mistakes.")

        dollar_amounts_text = st.text_area(
            "Dollar Amounts (one per line)",
            value="\n".join(data["dollar_amounts"]),
            height=150
        )

        percentages_text = st.text_area(
            "Percentages (one per line)",
            value="\n".join(data["percentages"]),
            height=100
        )

        scale_numbers_text = st.text_area(
            "Scale Numbers, e.g. users or market size (one per line)",
            value="\n".join(data["scale_numbers"]),
            height=100
        )

        team_mentions_text = st.text_area(
            "Team Mentions (one per line)",
            value="\n".join(data["team_mentions"]),
            height=100
        )

        if st.button("Save Corrections"):
            st.session_state["extracted_data"] = {
                "dollar_amounts": [line.strip() for line in dollar_amounts_text.split("\n") if line.strip()],
                "percentages": [line.strip() for line in percentages_text.split("\n") if line.strip()],
                "scale_numbers": [line.strip() for line in scale_numbers_text.split("\n") if line.strip()],
                "team_mentions": [line.strip() for line in team_mentions_text.split("\n") if line.strip()],
            }
            st.success("Corrections saved.")

elif page == "History":
    st.header("Startup History")
    st.write("Browse previously stored startup records from the database.")

    col1, col2 = st.columns([1, 3])
    with col1:
        page_number = st.number_input("Page", min_value=1, value=1, step=1)

    rows_per_page = 20
    skip = (page_number - 1) * rows_per_page

    with st.spinner("Loading records..."):
        history_response = requests.get(f"{API_BASE_URL}/startups", params={"skip": skip, "limit": rows_per_page})

    if history_response.status_code == 200:
        startups = history_response.json()

        if len(startups) == 0:
            st.info("No more records to show.")
        else:
            df = pd.DataFrame(startups)
            st.dataframe(df, use_container_width=True)
            st.caption(f"Showing records {skip + 1} to {skip + len(startups)}")
    else:
        st.error(f"Failed to load history: {history_response.text}")