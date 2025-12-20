import streamlit as st
from src.ml.predict import EnergySavingsPredictor

import os
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="ESG Energy Efficiency Agent", layout="wide")

# -----------------------------
# RAG Setup (loads Chroma DB from rag_project_artifacts/vector_store)
# -----------------------------
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

RAG_READY = True
if not os.environ.get("OPENAI_API_KEY"):
    RAG_READY = False

BASE_DIR = Path(__file__).resolve().parent
RAG_DB_DIR = BASE_DIR / "rag_project_artifacts" / "vector_store"

@st.cache_resource
def load_vectordb(persist_dir: Path):
    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings
    )

vectordb = None
if RAG_READY:
    if not RAG_DB_DIR.exists():
        RAG_READY = False
    else:
        vectordb = load_vectordb(RAG_DB_DIR)

@st.cache_resource
def load_llm():
    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)

SYSTEM_PROMPT = """
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

def rag_answer(question: str, k: int = 5) -> str:
    hits = vectordb.similarity_search(question, k=k)
    context = "\n\n".join(
        f"[SOURCE: {h.metadata.get('source')} | CHUNK: {h.metadata.get('chunk')}]\n{h.page_content}"
        for h in hits
    )
    prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"
    llm = load_llm()
    return llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]).content


# -----------------------------
# Single-page layout: ML first, then RAG
# -----------------------------
st.title("ESG Energy Efficiency Agent")
st.caption("Live demo — building inputs → ML prediction → ESG impact estimates")
st.caption("The model estimates post-intervention energy-savings potential by learning from patterns in retrofit outcomes of buildings with similar pre-intervention characteristics.")

# -----------------------------
# ML Predictor
# -----------------------------
@st.cache_resource
def load_predictor():
    return EnergySavingsPredictor()

predictor = load_predictor()

st.subheader("1) Building Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    floor_area_m2 = st.number_input("Floor area (m²)", 100.0, 200000.0, 5000.0, step=100.0)
    building_age_years = st.slider("Building age (years)", 0, 100, 25)

with col2:
    hvac_efficiency_score = st.slider("HVAC efficiency score", 0.4, 1.0, 0.7, 0.01)
    insulation_quality_score = st.slider("Insulation quality score", 0.3, 1.0, 0.6, 0.01)

with col3:
    occupancy_rate = st.slider("Occupancy rate", 0.5, 1.0, 0.85, 0.01)
    baseline_energy_kwh = st.number_input("Baseline annual energy (kWh)", 10000.0, 50_000_000.0, 750000.0, step=10000.0)

inputs = {
    "floor_area_m2": float(floor_area_m2),
    "building_age_years": float(building_age_years),
    "hvac_efficiency_score": float(hvac_efficiency_score),
    "insulation_quality_score": float(insulation_quality_score),
    "occupancy_rate": float(occupancy_rate),
    "baseline_energy_kwh": float(baseline_energy_kwh),
}

st.subheader("2) Prediction")

try:
    savings_pct = predictor.predict(inputs)
    st.success(f"Predicted energy savings: **{savings_pct:.2%}**")

    energy_saved_kwh = baseline_energy_kwh * savings_pct
    cost_saved = energy_saved_kwh * 0.25
    tco2_saved = (energy_saved_kwh * 0.7) / 1000

    c1, c2, c3 = st.columns(3)
    c1.metric("Energy saved (kWh/yr)", f"{energy_saved_kwh:,.0f}")
    c2.metric("Cost saved (AUD/yr)", f"${cost_saved:,.0f}")
    c3.metric("Emissions reduced (tCO₂e/yr)", f"{tco2_saved:,.1f}")

except Exception as e:
    st.error(f"Prediction failed: {e}")

st.divider()

# -----------------------------
# RAG Knowledge Base Assistant (same page, below ML)
# -----------------------------
st.header("📚 Ask the Knowledge Base (RAG)")
st.caption("Grounded answers with evidence from the project docs. Decision-support only. No certified ratings.")

if not RAG_READY:
    st.warning(
        "RAG is not ready.\n\n"
        "Fix checklist:\n"
        "1) Add OPENAI_API_KEY in Streamlit Secrets\n"
        "2) Ensure rag_project_artifacts/vector_store exists in this repo"
    )
else:
    q = st.text_input("Ask a question", placeholder="e.g., Does this tool give NABERS ratings?")
    ask_btn = st.button("Ask")

    if ask_btn and q.strip():
        with st.spinner("Retrieving evidence and generating answer..."):
            ans = rag_answer(q.strip())
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
