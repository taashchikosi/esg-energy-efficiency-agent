# app.py — FULL COPY/PASTE (ESG tab + NEW Healthcare tab)
# -------------------------------------------------------
# ✅ What you MUST do after pasting:
# 1) In requirements.txt add: requests
# 2) Paste your two GitHub Release asset links below (HEALTH_MODELS_URL / HEALTH_META_URL)
#    They must look like: .../releases/download/healthcare-v1/models.joblib
# 3) Keep your ESG files/folders exactly as-is:
#    - src/ml/predict.py  (EnergySavingsPredictor)
#    - rag_project_artifacts/vector_store  (Chroma persist dir)
#    - ESG_Energy_and_Emissions_Optimization_Agent.pdf  (repo root)

import os
import json
import joblib
import requests
from pathlib import Path

import pandas as pd
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.ml.predict import EnergySavingsPredictor


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Portfolio — Decision Support Agents", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# OpenAI API Key (for RAG)
# -----------------------------
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


# ============================================================
# ESG — RAG Setup (loads Chroma DB from rag_project_artifacts/vector_store)
# ============================================================
ESG_RAG_DB_DIR = BASE_DIR / "rag_project_artifacts" / "vector_store"

@st.cache_resource
def load_esg_vectordb(persist_dir: Path):
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings
    )

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)

ESG_SYSTEM_PROMPT = """
You are an energy-efficiency decision-support assistant.

Rules (must follow):
- Use ONLY the provided CONTEXT for factual claims.
- Do NOT invent facts or standards.
- If the CONTEXT is insufficient, say so and ask a specific follow-up question.
- Always include citations to the retrieved sources.

Return format:
1) Answer
2) Why
3) Evidence (bullets with source + chunk)
4) Confidence (High/Medium/Low)
5) Human Review Trigger (Yes/No + reason)
"""

def esg_rag_answer(vectordb, question: str, k: int = 5) -> str:
    hits = vectordb.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[SOURCE: {h.metadata.get('source')} | CHUNK: {h.metadata.get('chunk')}]\n{h.page_content}"
        for h in hits
    )
    prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
    llm = load_llm()
    return llm.invoke([
        {"role": "system", "content": ESG_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]).content


# ============================================================
# ESG — ML Predictor
# ============================================================
@st.cache_resource
def load_esg_predictor():
    return EnergySavingsPredictor()

esg_predictor = load_esg_predictor()


# ============================================================
# Healthcare — Auto-download model artifacts from GitHub Release
# ============================================================
HEALTH_ARTIFACTS_DIR = BASE_DIR / "healthcare_project_artifacts"
HEALTH_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

HEALTH_MODELS_PATH = HEALTH_ARTIFACTS_DIR / "models.joblib"
HEALTH_META_PATH   = HEALTH_ARTIFACTS_DIR / "model_meta.json"

# 🔴 PASTE YOUR *DIRECT* GITHUB RELEASE ASSET LINKS HERE:
# Example:
# HEALTH_MODELS_URL = "https://github.com/<user>/<repo>/releases/download/healthcare-v1/models.joblib"
# HEALTH_META_URL   = "https://github.com/<user>/<repo>/releases/download/healthcare-v1/model_meta.json"
HEALTH_MODELS_URL = "https://github.com/taashchikosi/esg-energy-efficiency-agent/releases/download/healthcare-v1/models.joblib"
HEALTH_META_URL   = "https://github.com/taashchikosi/esg-energy-efficiency-agent/releases/download/healthcare-v1/model_meta.json"

@st.cache_resource
def ensure_healthcare_artifacts():
    """
    Downloads the Healthcare artifacts once (cached by Streamlit).
    This makes the Healthcare tab WORK on Streamlit Cloud without committing large model files to the repo.
    """
    # meta first (small)
    if not HEALTH_META_PATH.exists():
        r = requests.get(HEALTH_META_URL, timeout=60)
        r.raise_for_status()
        HEALTH_META_PATH.write_bytes(r.content)

    # model (bigger)
    if not HEALTH_MODELS_PATH.exists():
        r = requests.get(HEALTH_MODELS_URL, timeout=240)
        r.raise_for_status()
        HEALTH_MODELS_PATH.write_bytes(r.content)

    return True


@st.cache_resource
def load_healthcare_artifacts():
    """
    Requires:
      - healthcare_project_artifacts/models.joblib
      - healthcare_project_artifacts/model_meta.json
    """
    models = joblib.load(HEALTH_MODELS_PATH)
    with open(HEALTH_META_PATH, "r") as f:
        meta = json.load(f)
    return models, meta


# ============================================================
# Healthcare — Scenario → Feature Builder (Streamlit-ready)
# ============================================================
ARRIVAL_PRESSURE_MAP = {"Low": 0.75, "Normal": 1.00, "High": 1.35}
STAFF_LEVEL_MAP      = {"Low": 0.75, "Normal": 1.00, "High": 1.25}

def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))

def build_health_feature_row(meta, capacity_beds, current_occupancy_pct, arrivals_pressure, staffing_level, current_wait_minutes, dt):
    FEATURES = meta["features"]

    occ_ratio = clamp(current_occupancy_pct / 100.0, 0.05, 1.08)

    # heuristic baseline arrivals scales with bed capacity (demo-friendly, consistent)
    base_arrivals = 6 + (capacity_beds / 700) * 19  # ~6..25
    arrivals_now = base_arrivals * ARRIVAL_PRESSURE_MAP[arrivals_pressure]

    staff_now = clamp(STAFF_LEVEL_MAP[staffing_level], 0.4, 1.6)
    wait_now  = clamp(current_wait_minutes, 5, 360)

    hour = int(dt.hour)
    dayofweek = int(dt.dayofweek)
    month = int(dt.month)
    is_weekend = 1.0 if dayofweek >= 5 else 0.0

    # conservative “last 24h” proxies for a demo (keeps UI simple)
    arrivals_roll_mean_24h = arrivals_now * clamp(0.95 + 0.05 * (1 if 12 <= hour <= 20 else -1), 0.85, 1.05)
    arrivals_roll_max_24h  = arrivals_now * (1.15 if arrivals_pressure != "Low" else 1.05)

    occ_roll_mean_24h = clamp(occ_ratio * (0.98 if is_weekend else 1.00), 0.05, 1.08)
    occ_roll_max_24h  = clamp(occ_ratio + (0.06 if arrivals_pressure == "High" else 0.03), 0.05, 1.08)

    wait_roll_mean_24h = wait_now * (1.05 if arrivals_pressure == "High" else 1.00)
    wait_roll_max_24h  = clamp(wait_now * (1.35 if arrivals_pressure == "High" else 1.15), 5, 360)

    row = {
        "capacity_beds": float(capacity_beds),
        "occupancy_ratio": float(occ_ratio),
        "arrivals": float(arrivals_now),
        "staff_index": float(staff_now),
        "wait_minutes": float(wait_now),
        "hour": float(hour),
        "dayofweek": float(dayofweek),
        "month": float(month),
        "is_weekend": float(is_weekend),
        "arrivals_roll_mean_24h": float(arrivals_roll_mean_24h),
        "arrivals_roll_max_24h": float(arrivals_roll_max_24h),
        "occ_roll_mean_24h": float(occ_roll_mean_24h),
        "occ_roll_max_24h": float(occ_roll_max_24h),
        "wait_roll_mean_24h": float(wait_roll_mean_24h),
        "wait_roll_max_24h": float(wait_roll_max_24h),
    }

    X = pd.DataFrame([row])[FEATURES]
    return X

def predict_healthcare(models, X):
    rf_occ = models["rf_occ"]
    rf_wait = models["rf_wait"]

    pred_max_occ_ratio_24h = float(rf_occ.predict(X)[0])
    pred_mean_wait_24h = float(rf_wait.predict(X)[0])

    # screening-level risk rule
    risk_high = int((pred_max_occ_ratio_24h >= 0.95) or (pred_mean_wait_24h >= 120))
    return pred_max_occ_ratio_24h, pred_mean_wait_24h, risk_high


# ============================================================
# App Layout — Tabs
# ============================================================
st.title("AI Portfolio — Decision Support Agents")
st.caption("Select a project tab below.")

tab_esg, tab_health = st.tabs([
    "🌱 ESG Energy & Emissions Optimization Agent",
    "🏥 Healthcare Patient-Flow Optimization Agent"
])


# ============================================================
# TAB 1 — ESG (your original app, preserved)
# ============================================================
with tab_esg:
    st.title("ESG Energy & Emissions Optimization Agent")
    st.subheader("AI Decision-Support Tool for Building Retrofit Prioritization")

    st.caption(
        "This tool estimates post-intervention energy-savings, cost-savings, and emissions reduction potential "
        "by learning from patterns in retrofit outcomes of buildings with similar pre-intervention characteristics. "
        "Its main goal is to help decision-makers decide where to spend limited retrofit capital first, and how confidently they can justify that choice."
    )

    st.subheader("Examples of Decisions it Supports")
    st.caption("1. Prioritization Decisions - Which buildings have the highest savings potential?")
    st.caption("2. Capital Allocation Decisions - Where should we deploy limited budget or resources?")
    st.caption("3. Scenario Comparison Decisions - How sensitive are outcomes to changes in building characteristics?")
    st.caption("4. Governance & Denfensibility Decisions - Can we responsibly communicate this insight to executives, investors, or ESG reports?")

    st.subheader("How to use this tool - The Decision Workflow")
    st.caption("1. Input baseline characteristics -> this defines the current state of the building")
    st.caption("2. Allow the machine to predict potential ESG outcomes -> Review estimated energy, cost, & emissions impacts.")
    st.caption("3. Use the Assistant to ask questions -> Interogate the rationale, understand assumptions, limitations, & governance implications.")
    st.caption("4. Decide next action —> investigate further, prioritise or deprioritize buildings")

    # ML Predictor
    st.subheader("1) Building Baseline")
    st.caption("Describe the current (pre-retrofit) state of the building. These inputs define the baseline against which potential improvements are estimated.")

    col1, col2, col3 = st.columns(3)

    with col1:
        floor_area_m2 = st.number_input("Gross Floor area (m²)", 100.0, 200000.0, 5000.0, step=100.0)
        building_age_years = st.slider("Building Age (years since construction)", 0, 100, 25)

    with col2:
        hvac_efficiency_score = st.slider("HVAC Efficiency (relative score)", 0.4, 1.0, 0.7, 0.01)
        insulation_quality_score = st.slider("Envelope/Insulation quality (relative score)", 0.3, 1.0, 0.6, 0.01)

    with col3:
        occupancy_rate = st.slider("Average Occupancy Utilisation %", 0.5, 1.0, 0.85, 0.01)
        baseline_energy_kwh = st.number_input("Baseline annual energy consumption (kWh)", 10000.0, 50_000_000.0, 750000.0, step=10000.0)

    inputs = {
        "floor_area_m2": float(floor_area_m2),
        "building_age_years": float(building_age_years),
        "hvac_efficiency_score": float(hvac_efficiency_score),
        "insulation_quality_score": float(insulation_quality_score),
        "occupancy_rate": float(occupancy_rate),
        "baseline_energy_kwh": float(baseline_energy_kwh),
    }

    st.subheader("2) Estimated Post-Intervention Impact")
    st.caption("Estimated outcomes assuming representative retrofit interventions applied to similar buildings.")

    try:
        savings_pct = esg_predictor.predict(inputs)
        st.success(f"Predicted energy savings: **{savings_pct:.2%}**")

        energy_saved_kwh = baseline_energy_kwh * savings_pct
        cost_saved = energy_saved_kwh * 0.25
        tco2_saved = (energy_saved_kwh * 0.7) / 1000

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated annual energy reduction (kWh/year)", f"{energy_saved_kwh:,.0f}")
        c2.metric("Indicative operating cost reduction (AUD/year)", f"${cost_saved:,.0f}")
        c3.metric("Estimated Scope 2 emissions reduction (tCO₂e/yr)", f"{tco2_saved:,.1f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

    st.divider()

    # RAG Knowledge Base Assistant
    st.header("🧠 Decision Rationale & Evidence Assistant")
    st.caption("I provide evidence-grounded explanations, limitations, and governance context to support responsible ESG-aligned investment and prioritization decision-making. Ask me anything!")

    ESG_RAG_READY = bool(os.environ.get("OPENAI_API_KEY")) and ESG_RAG_DB_DIR.exists()

    if not ESG_RAG_READY:
        st.warning(
            "RAG is not ready.\n\n"
            "Fix checklist:\n"
            "1) Add OPENAI_API_KEY in Streamlit Secrets\n"
            "2) Ensure rag_project_artifacts/vector_store exists in this repo"
        )
    else:
        vectordb = load_esg_vectordb(ESG_RAG_DB_DIR)
        q = st.text_input("Ask a question", key="esg_q", placeholder="e.g., Does this tool give NABERS ratings?")
        ask_btn = st.button("Ask", key="esg_ask")

        if ask_btn and q.strip():
            with st.spinner("Retrieving evidence and generating answer..."):
                ans = esg_rag_answer(vectordb, q.strip())
            st.markdown("### Response")
            st.write(ans)

        with st.expander("Suggested demo questions"):
            st.write(
                "1) What decisions does this tool support?\n"
                "2) Does this tool give NABERS ratings?\n"
                "3) How do we avoid greenwashing when communicating results?\n"
                "4) What are the model limitations?\n"
                "5) When should human expert review be triggered?\n"
            )

    st.caption("This demo uses synthetic and simulation-informed data to demonstrate decision-support workflows. Outputs are directional and intended for prioritisation, not certification.")

    # ESG PDF Download
    st.markdown("---")
    st.markdown("## 📄 Full Consulting Report")

    esg_pdf_file = "ESG_Energy_and_Emissions_Optimization_Agent.pdf"
    try:
        with open(esg_pdf_file, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="⬇️ Download Report (PDF)",
            data=pdf_bytes,
            file_name=esg_pdf_file,
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.info(
            f"PDF not found: {esg_pdf_file}\n\n"
            "Upload it to the repo root (same folder as app.py)."
        )


# ============================================================
# TAB 2 — Healthcare (NEW)
# ============================================================
with tab_health:
    st.title("Healthcare Patient-Flow Optimization Agent")
    st.subheader("AI Decision-Support Tool for Next-24h Congestion Risk Screening")

    st.caption(
        "This tool forecasts near-term operational pressure (next 24 hours) from the hospital’s current state. "
        "It is designed for screening and prioritization (early warning), not automated staffing or clinical decisions."
    )

    # Download artifacts from GitHub Release (so the tab actually works)
    try:
        # If you forgot to paste URLs, fail with a clear message
        if "PASTE_" in HEALTH_MODELS_URL or "PASTE_" in HEALTH_META_URL:
            st.error("Healthcare Release URLs are not set. Paste the two GitHub Release asset links into HEALTH_MODELS_URL and HEALTH_META_URL at the top of app.py.")
            st.stop()

        ensure_healthcare_artifacts()
    except Exception as e:
        st.error(f"Failed to download Healthcare model artifacts: {e}")
        st.stop()

    # Load artifacts
    try:
        models, meta = load_healthcare_artifacts()
    except Exception as e:
        st.error(f"Failed to load Healthcare model artifacts after download: {e}")
        st.stop()

    st.subheader("1) Current Operational State (Inputs)")
    st.caption("Set a high-level snapshot of current conditions. The app constructs a valid feature vector behind the scenes.")

    c1, c2 = st.columns(2)

    with c1:
        capacity_beds = st.slider("Hospital bed capacity", 150, 700, 450, step=10)
        current_occupancy_pct = st.slider("Current occupancy (%)", 10, 108, 85, step=1)
        arrivals_pressure = st.selectbox("Recent arrivals pressure", ["Low", "Normal", "High"], index=1)

    with c2:
        staffing_level = st.selectbox("Staffing level (relative)", ["Low", "Normal", "High"], index=1)
        current_wait_minutes = st.slider("Current average wait time (minutes)", 5, 240, 60, step=5)
        date = st.date_input("Date", key="health_date")

    hour = st.slider("Hour of day", 0, 23, 18, step=1, key="health_hour")
    dt = pd.Timestamp(date) + pd.Timedelta(hours=hour)

    st.subheader("2) Next-24h Forecast (Outputs)")
    st.caption("Screening-level signals to support earlier operational attention and prioritization.")

    if st.button("Run 24-hour forecast", key="health_run"):
        X = build_health_feature_row(
            meta=meta,
            capacity_beds=capacity_beds,
            current_occupancy_pct=current_occupancy_pct,
            arrivals_pressure=arrivals_pressure,
            staffing_level=staffing_level,
            current_wait_minutes=current_wait_minutes,
            dt=dt
        )

        pred_max_occ, pred_mean_wait, risk_high = predict_healthcare(models, X)

        k1, k2, k3 = st.columns(3)
        k1.metric("Predicted max occupancy ratio (next 24h)", f"{pred_max_occ:.2f}")
        k2.metric("Predicted mean wait (minutes, next 24h)", f"{pred_mean_wait:.0f}")
        k3.metric("Risk flag (screening)", "HIGH" if risk_high else "LOW")

        if risk_high:
            st.error("High congestion risk (screening signal). Use as an early-warning prompt for human review, not as an automated instruction.")
        else:
            st.success("Lower congestion risk (screening signal). Continue monitoring and reassess if conditions change.")

        with st.expander("Show constructed model inputs (feature row)"):
            st.dataframe(X)

    st.markdown("---")
    st.markdown("## 📄 Full Consulting Report (Healthcare)")

    health_pdf_file = "Healthcare_Patient_Flow_Optimization_Agent.pdf"
    try:
        with open(health_pdf_file, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            label="⬇️ Download Report (PDF)",
            data=pdf_bytes,
            file_name=health_pdf_file,
            mime="application/pdf",
        )
    except FileNotFoundError:
        st.info(
            f"PDF not found: {health_pdf_file}\n\n"
            "When you generate it at the end, upload it to the repo root (same folder as app.py)."
        )

    st.caption(
        "Governance note: This is operational decision support for screening/prioritization. "
        "It does not provide medical advice, staffing schedules, or clinical decisions."
    )
