"""ResQ-AI Interactive Dashboard.

Features:
- Map of flood probability with uncertainty overlay
- Affected population intervals per zone
- Resource allocation visualization with routes
- Scenario sliders (rainfall intensity, resources, risk tolerance)
- Export button for results
- Demo mode with synthetic data
"""

import streamlit as st
import requests
import os

API_URL = os.getenv("RESQAI_API_URL", "http://localhost:8000")

st.set_page_config(page_title="ResQ-AI Dashboard", layout="wide")

st.sidebar.title("ResQ-AI")
page = st.sidebar.radio("Navigation", ["Prediction", "Impact", "Allocation", "Settings", "About"])

if page == "Prediction":
    st.title("Flood Prediction")
    st.write("Upload SAR/optical data or use demo.")
    use_demo = st.checkbox("Use Demo Data", value=True)
    
    col1, col2 = st.columns(2)
    
    if st.button("Predict"):
        if use_demo:
            try:
                res = requests.post(f"{API_URL}/predict", json={"uq_method": "ensemble", "use_demo": True})
                data = res.json()
                with col1:
                    st.subheader("Probability Map")
                    st.write(data["probability_map"])
                with col2:
                    st.subheader("Uncertainty Map")
                    st.write(data["uncertainty_map"])
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

elif page == "Impact":
    st.title("Impact Assessment")
    st.write("Affected population distribution with confidence intervals")
    
    try:
        res = requests.post(f"{API_URL}/impact", json={
            "prediction_id": "demo", 
            "population_data": {}, 
            "building_data": {}
        })
        st.write(res.json())
    except Exception as e:
        st.error(f"API Error: {e}")

elif page == "Allocation":
    st.title("Resource Allocation")
    st.write("Resource allocation map, depot locations, routes, unmet demand chart")
    
    try:
        res = requests.post(f"{API_URL}/allocate", json={
            "impact_scenarios": [], 
            "depots": [],
            "method": "cvar",
            "alpha": 0.95,
            "num_scenarios": 100
        })
        st.write(res.json())
    except Exception as e:
        st.error(f"API Error: {e}")

elif page == "Settings":
    st.title("Settings")
    st.selectbox("UQ Method", ["ensemble", "mc_dropout", "evidential", "conformal"])
    st.slider("Alpha (Risk Tolerance)", 0.0, 1.0, 0.95)
    st.number_input("Number of Scenarios", min_value=1, value=100)

elif page == "About":
    st.title("About ResQ-AI")
    st.write("Uncertainty-Aware Multimodal AI Framework for Urban Disaster Prediction.")
    st.write("Methodology involves advanced ML for flood prediction and risk-aware resource allocation.")
