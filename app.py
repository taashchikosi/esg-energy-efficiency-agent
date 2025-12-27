import streamlit as st
from src.ml.predict import EnergySavingsPredictor

import os
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="ESG Energy & Emissions Optimization Agent", layout="wide")

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
st.title("ESG Energy & Emissions Optimization Agent")
st.subheader("AI Decision-Support Tool for Building Retrofit Prioritization")

st.caption("This tool estimates post-intervention energy-savings, cost-savings, and emissions reduction potential by learning from patterns in retrofit outcomes of buildings with similar pre-intervention characteristics as below. It's main goal is to help decision-makers decide where to spend limited retrofit capital first, and how confidently they can justify that choice.")

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

# -----------------------------
# ML Predictor
# -----------------------------
@st.cache_resource
def load_predictor():
    return EnergySavingsPredictor()

predictor = load_predictor()

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
    savings_pct = predictor.predict(inputs)
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

# -----------------------------
# RAG Knowledge Base Assistant (same page, below ML)
# -----------------------------
st.header("🧠 Decision Rationale & Evidence Assistant")
st.caption("I provide evidence-grounded explanations, limitations, and governance context to support responsible ESG-aligned investment and prioritization decision-making. Ask me anything!")

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

st.caption("This demo uses synthetic and simulation-informed data to demonstrate decision-support workflows. Outputs are directional and intended for prioritisation, not certification.")

# ===============================
# Full Consulting Report (PDF)
# ===============================

import base64

st.markdown("---")
st.markdown("## 📄 Full Consulting Report")

st.write(
    "The full consulting-grade report supporting this decision-support tool "
    "is available below. It documents the decision context, data strategy, "
    "modeling approach, governance design, and deployment scope."
)

with open("ESG_Energy_and_Emissions_Optimization_Agent.pdf", "rb") as f:
    pdf_bytes = f.read()

base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

pdf_display = f"""
<iframe 
    src="data:application/pdf;base64,{base64_pdf}" 
    width="100%" 
    height="900px" 
    style="border: none;">
</iframe>
"""

with st.expander("📘 View Full Report"):
    st.markdown(pdf_display, unsafe_allow_html=True)

