import streamlit as st
from src.ml.predict import EnergySavingsPredictor

st.set_page_config(page_title="ESG Energy Efficiency Agent", layout="wide")

st.title("ESG Energy Efficiency Agent")
st.caption("Live demo — building inputs → ML prediction → ESG impact estimates")
st.caption("The model estimates post-intervention energy-savings potential by learning from patterns in retrofit outcomes of buildings with similar pre-intervention characteristics.")

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
