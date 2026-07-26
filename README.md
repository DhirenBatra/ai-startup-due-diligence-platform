# AI-DDP — AI-Powered Startup Due Diligence Platform

> End-to-end startup evaluation pipeline: ML success prediction · SHAP explainability · LLM report generation · PDF/OCR parsing · FastAPI + Streamlit

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.140-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.60-red)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.9-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-brightgreen)
![SHAP](https://img.shields.io/badge/SHAP-0.51-yellow)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED)

**[Live Dashboard](https://ai-startup-due-diligence-platform-1.onrender.com)** · **[API Docs](https://ai-startup-due-diligence-platform.onrender.com/docs)**

---

## Overview

AI-DDP is a full-stack due-diligence system built on real startup outcome data. It predicts a startup's probability of success using a tuned Random Forest model, explains that prediction feature-by-feature with SHAP, and hands both off to an LLM that writes a structured four-part investment report. A separate OCR-backed pipeline reads pitch deck PDFs directly — including fully image-based decks — and pulls out funding figures, market stats, and team information automatically.

**Key Results:**

| Metric | Value |
|---|---|
| Startups in dataset | 923 (Kaggle Startup Success Prediction) |
| Features after cleaning | 33 (+5 engineered) |
| Best model | Random Forest (tuned) |
| Test ROC-AUC | 0.840 |
| Test Accuracy | 0.789 (untuned baseline) |
| YC records enriched | 8 verified matches (of 17 raw, 9 filtered as false positives) |
| API endpoints | 5 (FastAPI, fully documented at `/docs`) |
| Deployment | 2 live Render services (backend + frontend) |
| PDF OCR test case | Real 14-slide pitch deck, single image-based page |

---

## Architecture

```
ai-startup-due-diligence-platform/
├── src/
│   ├── main.py                  # FastAPI app — all 5 endpoints
│   ├── database.py              # SQLAlchemy models (startups, scores, reports)
│   ├── create_tables.py         # One-time table creation
│   ├── load_data_to_db.py       # Loads cleaned CSV into Postgres
│   ├── test_db_connection.py    # DB connectivity check
│   ├── pdf_parser.py            # pdfplumber + PyMuPDF + Tesseract OCR + regex extraction
│   ├── generate_report.py       # Standalone LLM report generator
│   ├── clean_data.py            # Dataset cleaning (49 → 33 columns)
│   ├── merge_yc_data.py         # Y Combinator enrichment + batch-year filter
│   ├── check_merged_data.py     # Merge verification
│   ├── feature_engineering.py   # 5 derived features
│   ├── eda.py                   # Correlation analysis + charts
│   ├── train_model.py           # Logistic Regression / Random Forest / XGBoost comparison
│   ├── tune_and_explain.py      # RandomizedSearchCV + SHAP explainer
│   └── load_and_explore.py      # Initial dataset inspection
├── streamlit_app.py             # Frontend — Predict / Upload / History pages
├── data/                        # Raw + cleaned CSVs, sample pitch deck PDF
├── models/                      # best_model.pkl · scaler.pkl · shap_explainer.pkl · feature_columns.pkl
├── reports/                     # EDA charts, SHAP plots, data dictionary, sample outputs
├── .streamlit/config.toml       # Custom dashboard theme
├── Dockerfile.backend           # FastAPI image (includes Tesseract OCR)
├── Dockerfile.frontend          # Streamlit image
├── docker-compose.yml           # Local multi-container orchestration
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| ML Models | Scikit-learn (Logistic Regression, Random Forest) · XGBoost |
| Explainability | SHAP (TreeExplainer) |
| Generative AI | OpenRouter (`openrouter/free`) — OpenAI-compatible client |
| PDF Parsing | pdfplumber · PyMuPDF (fitz) · Tesseract OCR (pytesseract) |
| Frontend | Streamlit + Plotly |
| Database | PostgreSQL (Supabase) via SQLAlchemy |
| Containerization | Docker, Docker Compose |
| Deployment | Render (2 independent Web Services) |

---

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL database (a free [Supabase](https://supabase.com) project works well)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed locally
- An [OpenRouter](https://openrouter.ai) API key

### 1. Clone and install dependencies

```bash
git clone https://github.com/DhirenBatra/ai-startup-due-diligence-platform.git
cd ai-startup-due-diligence-platform
python -m venv venv
venv\Scripts\activate        # Windows — use source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```
DATABASE_URL=postgresql://your-connection-string
OPENROUTER_API_KEY=your-key
```

### 3. Set up the database

```bash
cd src
python create_tables.py
python load_data_to_db.py
cd ..
```

### 4. Run

```bash
uvicorn src.main:app --reload          # Terminal 1 — backend
streamlit run streamlit_app.py         # Terminal 2 — frontend
```

- Frontend: `http://localhost:8501`
- Backend API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

---

## Requirements

Key packages in `requirements.txt`:

```
fastapi
uvicorn
sqlalchemy
psycopg2-binary
python-multipart
scikit-learn
xgboost
shap
pandas
numpy
joblib
pdfplumber
PyMuPDF
pytesseract
Pillow
openai
python-dotenv
streamlit
plotly
```

---

## Usage

### Run the full pipeline standalone (no API)

```bash
python src/train_model.py
python src/tune_and_explain.py
python src/generate_report.py
```

### Test the API directly

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age_first_funding_year": 1.5, "funding_rounds": 3, "relationships": 8, ...}'
```

Or use the interactive Swagger UI at `/docs` — every endpoint can be tested from the browser.

### Run with Docker

```bash
docker-compose up --build
```

Starts backend (port 8000) and frontend (port 8501) together. Inside Compose, the frontend reaches the backend at `http://backend:8000`; outside Docker it defaults to `127.0.0.1:8000` via the `API_BASE_URL` environment variable.

---

## Pipeline Flow

```
Raw Kaggle CSV (923 rows, 49 cols)
↓
clean_data.py — drop redundant cols · fill missing milestones → 33 cols
↓
merge_yc_data.py — YC enrichment + batch-year plausibility filter → 8 trustworthy matches
↓
eda.py — correlation analysis (relationships, milestones ↑ · is_otherstate, has_VC ↓)
↓
feature_engineering.py — +5 derived features (funding_velocity, milestone_rate, etc.) → 38 cols
↓
train_model.py — Logistic Regression (scaled) · Random Forest · XGBoost
↓
Best model selected by ROC-AUC → Random Forest
↓
tune_and_explain.py — RandomizedSearchCV (5-fold CV) → ROC-AUC 0.840 · SHAP TreeExplainer fit
↓
FastAPI /predict — serves probability + top-5 SHAP factors
↓
FastAPI /generate-report — LLM (OpenRouter) writes 4-section report from probability + SHAP factors
↓
FastAPI /upload-pitch-deck — pdfplumber → OCR fallback (Tesseract) → regex extraction
↓
Streamlit dashboard — gauge chart · SHAP bar chart · colored report panels · history table
```

---

## Model Training & Explainability

**Models compared** (80/20 stratified split):

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression (scaled) | 0.735 | 0.793 | 0.810 |
| Random Forest | 0.789 | 0.843 | 0.835 |
| XGBoost | 0.773 | 0.832 | 0.809 |
| **Random Forest (tuned)** | — | — | **0.840** |

**Tuning:** `RandomizedSearchCV`, 20 iterations, 5-fold CV, scored on ROC-AUC. Best params: 300 estimators, max depth 10, min split 5, min leaf 4.

**Key bug fixed:** unscaled Logistic Regression collapsed to predicting a single class because `funding_total_usd` (values in the hundreds of thousands) dominated small binary flags like `has_VC` in the loss function. `StandardScaler` (fit on train only) raised accuracy from 65.4% → 73.5%.

**Explainability:** every prediction returns its top 5 SHAP factors and their direction (positive/negative), computed with a `TreeExplainer` against the tuned Random Forest.

---

## LLM Report Generation

**Provider journey:** the spec called for OpenAI (paid). Gemini was tried first (free-tier quota stayed at zero even on a fresh project; billing setup pushed an unrelated ₹15k/year plan). DeepSeek was tried next (connected, but returned `Insufficient Balance`). **OpenRouter's free router** (`openrouter/free`) was the working solution — it load-balances across multiple free models instead of depending on one.

**Prompt design:** fixed 4-section template (`## Risk Summary`, `## Strengths`, `## Red Flags`, `## Recommendation`), populated with the startup's raw data, ML probability, and SHAP factors — grounding the narrative in the same numbers shown in the chart.

**Retry logic:** any response under 200 characters (a sign a free-tier model returned a degenerate reply) is automatically retried once.

---

## PDF Parsing & OCR

**Pipeline:** `pdfplumber` first → if a page yields under 100 characters, treat it as image-based → render with PyMuPDF → OCR with Tesseract → regex-extract dollar amounts, percentages, scale numbers, and team mentions (`"Name | Title"` pattern).

**Real test case:** ElevenLabs' 2022–23 pre-seed deck — all 14 slides exported as a single, very tall image-based page. Direct extraction returned 23 characters; OCR fallback correctly pulled $2B, $4.6B, $110M, 96M/50M user figures, and both team members with their titles.

**Production constraint:** this same page crashed Render's free-tier backend (512MB RAM, exit code 137 / OOM). Fixed by processing the page in horizontal grayscale strips instead of one large image, freeing memory between strips — trades speed (5–7 min) for staying within the memory limit.

---

## Dataset

Real startup outcome data, enriched with public accelerator records.

- **Primary:** [Startup Success Prediction](https://www.kaggle.com/) (Kaggle, Manish KC) — 923 US startups, target column `labels`
- **Supplementary:** Y Combinator company metadata via the [yc-oss API](https://yc-oss.github.io/api/companies/all.json) (6,006 companies), merged by name with a batch-year plausibility filter to remove false-positive matches
- **PDF validation set:** ElevenLabs' real, public pre-seed pitch deck

---

## Database Schema

| Table | Purpose |
|---|---|
| `startups` | Full 32-feature record per company |
| `scores` | Saved ML predictions, linked via `startup_id` (FK) |
| `reports` | Saved LLM-generated narratives, linked via `startup_id` (FK) |

---

## Deployment

Backend and frontend are deployed as **two independent Render Web Services**, each built from its own Dockerfile against this repository. Render's free tier does not run `docker-compose.yml` directly, so each service is configured separately with its own environment variables — the frontend's `API_BASE_URL` points at the backend's live public address rather than the Docker-internal hostname used locally.

---

## What's Not Production-Ready (Yet)

- Training data is 923 startups, mostly pre-2014 — not current or large-scale
- Financials come from parsed pitch decks, not audited statements
- Free-tier infra (LLM router, hosting, database) means rate limits and cold starts
- No API authentication or rate limiting implemented
- `/predict` and `/generate-report` don't yet persist results back to `scores`/`reports` (schema supports it, endpoints don't write yet)

---

## Author

| Name | Institution | Scope |
|---|---|---|
| Dhiren Batra | UPES Dehradun, B.Tech CSE | Full pipeline — data, ML, backend, frontend, deployment |

---

## License

Academic project — UPES Dehradun, 2026.
