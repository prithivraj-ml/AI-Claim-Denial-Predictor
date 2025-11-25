import streamlit as st
import pandas as pd
import requests
import json
import shap
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
API_ENDPOINT = "http://127.0.0.1:5000" # Default Flask address

# --- Streamlit Setup ---
st.set_page_config(page_title="MedBillingAI Claim Predictor", layout="wide")
st.title("AI Claim Denial Predictor Dashboard")
st.markdown("Use this interface to predict the denial probability and understand the risk factors for a new claim.")

# --- Helper Function to Create a Simple Data Form ---
def create_claim_form():
    """Generates a sample dictionary of required claim features for input."""
    # NOTE: These columns must match the columns used in your training data
    
    st.header("1. Claim Input Data")

    # Define features based on the dataset generator for consistency
    ICD10_CODES = ['I10', 'E11', 'J45', 'M54', 'K21', 'N39', 'F32', 'G43', 'L40', 'E66']
    CPT_CODES = ['99213', '99214', '99203', '99204', '93000', '80053', '36415', '71020', '84443', '81002']
    PROVIDER_TYPES = ['Hospital', 'Clinic', 'Telemedicine']
    PROVIDER_SPECIALTY = ['Cardiology', 'Orthopedic', 'Radiology', 'General', 'Pediatrics', 'Oncology']
    INSURANCE_TYPES = ['Private', 'Medicare', 'Medicaid']
    STATES = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']

    cols = st.columns(3)

    # Column 1: Core Financial and Patient Data
    with cols[0]:
        st.subheader("Core Claim Details")
        claim_amount = st.slider("Claim Amount ($):", 50.0, 15000.0, 2500.0, 50.0)
        patient_age = st.slider("Patient Age:", 0, 90, 45)
        gender = st.selectbox("Gender:", ['M', 'F'])
        num_procedures = st.slider("Number of Procedures:", 1, 10, 3)
        num_diagnoses = st.slider("Number of Diagnoses:", 1, 5, 2)
        processing_days = st.slider("Processing Days:", 1, 30, 5)

    # Column 2: Service and Provider Data
    with cols[1]:
        st.subheader("Codes and Providers")
        icd_code = st.selectbox("ICD10 Code:", ICD10_CODES, index=0)
        cpt_code = st.selectbox("CPT Code:", CPT_CODES, index=1)
        provider_type = st.selectbox("Provider Type:", PROVIDER_TYPES)
        provider_specialty = st.selectbox("Provider Specialty:", PROVIDER_SPECIALTY)
        state = st.selectbox("State of Service:", STATES, index=0)
        
    # Column 3: Flags and Authorization
    with cols[2]:
        st.subheader("Risk and Status Flags")
        insurance_type = st.selectbox("Insurance Type:", INSURANCE_TYPES)
        prior_auth = st.selectbox("Prior Authorization Required:", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        telehealth = st.selectbox("Telehealth Service:", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        in_network = st.selectbox("In-Network Status:", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        chronic_flag = st.selectbox("Chronic Condition Flag:", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        high_risk_flag = st.selectbox("High Risk Flag (Manual Review):", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        resubmission_count = st.slider("Resubmission Count:", 0, 5, 0)
        
    # Fixed non-interactive features (still required by the model)
    submission_date = "2024-11-20"
    claim_id = st.text_input("Claim ID (for tracking):", "New_Claim_1")

    # --- CRITICAL: Ensure all variables are correctly assigned and match the API's keys ---
    # The variable names must be identical to the keys in INPUT_FEATURE_ORDER in api.py
    claim_data = {
        "Claim_ID": claim_id,
        "Patient_Age": patient_age,
        "Gender": gender,
        "ICD10_Code": icd_code, # Key is ICD10_Code, variable is icd_code
        "CPT_Code": cpt_code,   # Key is CPT_Code, variable is cpt_code
        "Provider_Type": provider_type,
        "Provider_Specialty": provider_specialty,
        "Insurance_Type": insurance_type,
        "State": state,
        "Claim_Amount": claim_amount,
        "Prior_Authorization": prior_auth,
        "Num_Procedures": num_procedures,
        "Num_Diagnoses": num_diagnoses,
        "Chronic_Flag": chronic_flag,
        "High_Risk_Flag": high_risk_flag,
        "Telehealth": telehealth,
        "In_Network": in_network,
        "Submission_Date": submission_date,
        "Processing_Days": processing_days,
        "Resubmission_Count": resubmission_count
    }
    
    st.markdown("---")
    return claim_data

# --- Prediction and Explanation Logic (Unchanged) ---

def get_prediction(data):
    """Calls the Flask /predict endpoint."""
    response = requests.post(f"{API_ENDPOINT}/predict", json=data)
    if response.status_code == 200:
        return response.json()
    # Updated error handling to show the error message from the API
    error_message = response.json().get('error', 'Unknown Error')
    st.error(f"Prediction API Error: {response.status_code} - {error_message}")
    return None

def get_explanation(data):
    """Calls the Flask /explain endpoint to get SHAP values."""
    response = requests.post(f"{API_ENDPOINT}/explain", json=data)
    if response.status_code == 200:
        return response.json()
    # Updated error handling to show the error message from the API
    error_message = response.json().get('error', 'Unknown Error')
    st.error(f"Explanation API Error: {response.status_code} - {error_message}")
    return None

# --- Main Dashboard Layout (Unchanged) ---

claim_data = create_claim_form()

if st.button("Predict Claim Denial Risk", type="primary"):
    
    st.subheader("2. Model Prediction")
    
    # 1. Get Prediction
    pred_result = get_prediction(claim_data)
    
    if pred_result:
        
        prob = pred_result['Denial_Probability']
        
        # Determine the visual representation based on threshold
        if prob >= pred_result['Threshold_Applied']:
            status_text = f"🚨 DENIAL RISK FLAGGED"
            status_color = "red"
        else:
            status_text = f"✅ ACCEPTED (Cleared)"
            status_color = "green"

        # Display Prediction Metrics
        st.markdown(f"**Final Status:** <span style='color:{status_color}; font-size: 24px;'>{status_text}</span>", unsafe_allow_html=True)
        
        st.metric(label="Denial Probability Score", 
                  value=f"{prob:.4f}", 
                  help=f"Custom Threshold: {pred_result['Threshold_Applied']:.4f}")

        st.markdown("---")

        # 2. Get Explanation (SHAP)
        st.subheader("3. Model Explanation (SHAP Waterfall)")
        
        explanation_result = get_explanation(claim_data)
        
        if explanation_result:
            
            # Prepare data for SHAP visualization
            shap_values = np.array(explanation_result['shap_values'])
            feature_names = explanation_result['feature_names']
            base_value = explanation_result['base_value']

            # Since the SHAP values are for the one-hot encoded features,
            # we need to create a dummy Explanation object for the waterfall plot.
            
            # We filter for the top 10 most impactful features for cleaner visualization
            top_indices = np.argsort(np.abs(shap_values))[-10:]
            
            # --- CRITICAL FIX FOR KEYERROR: 'Days' ---
            # We use a robust logic to infer the original key from the transformed feature name:
            # 1. If the full transformed name (f) is a key in claim_data (e.g., 'Patient_Age', 'Processing_Days'), use it.
            # 2. Otherwise, assume it is an OHE feature (e.g., 'ICD10_Code_I10') and split it to get the original key (e.g., 'ICD10_Code').
            data_for_plot = np.array([
                claim_data[
                    f if f in claim_data else 
                    '_'.join(f.split('_')[:-1])
                ] for f in np.array(feature_names)[top_indices]
            ])
            # ----------------------------------------
            
            # Create the Explanation object
            explanation = shap.Explanation(
                values=shap_values[top_indices],
                base_values=base_value,
                data=data_for_plot, # Use the correctly generated data array
                feature_names=np.array(feature_names)[top_indices]
            )
            
            # Generate and display the waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig, clear_figure=True)
            st.caption("The base value is the average denial probability. Features pushing the score to the right (red) increase denial risk.")