import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import shap
from sklearn.compose import ColumnTransformer # Needed for type hints

# --- CONFIGURATION ---
# IMPORTANT: SET YOUR OPTIMIZED THRESHOLD HERE
# This should be the value (e.4725) that gave you the highest Recall/F1-Score
DECISION_THRESHOLD = 0.4725 

# --- CRITICAL: Define the exact feature order used during model training ---
# This list MUST match the order of columns fed into the ColumnTransformer during training.
INPUT_FEATURE_ORDER = [
    'Claim_ID', 'Patient_Age', 'Gender', 'ICD10_Code', 'CPT_Code', 
    'Provider_Type', 'Provider_Specialty', 'Insurance_Type', 'State', 
    'Claim_Amount', 'Prior_Authorization', 'Num_Procedures', 'Num_Diagnoses', 
    'Chronic_Flag', 'High_Risk_Flag', 'Telehealth', 'In_Network', 
    'Submission_Date', 'Processing_Days', 'Resubmission_Count'
]

# --- UTILITY FUNCTION FOR FEATURE NAMES ---
# This is required because the OneHotEncoder obscures the original column names
def get_feature_names(column_transformer: ColumnTransformer):
    """Retrieves feature names after ColumnTransformer processing."""
    feature_names = []
    for name, transformer, original_features in column_transformer.transformers_:
        if name == 'remainder':
            continue
        
        # Check if the transformer has get_feature_names_out (modern way)
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(original_features))
        elif name == 'num': 
            feature_names.extend(original_features)
    return feature_names

# ------------------------------------------
# 1. LOAD MODEL + PREPROCESSOR + SHAP INIT
# ------------------------------------------
try:
    # 'model' is the full pipeline (preprocessor + model)
    model = joblib.load("model_xgb_optimized.pkl") 
    
    # 'preprocessor' is the ColumnTransformer step
    preprocessor = model.named_steps["preprocess"]
    
    # 'xgb_model' is the final XGBoost model step
    xgb_model = model.named_steps["model"]
    
    # explainer is initialized ONLY on the raw XGBoost model
    explainer = shap.TreeExplainer(xgb_model)
    
    # Get the feature names for the explain endpoint
    FEATURE_NAMES = get_feature_names(preprocessor)
    print("✅ Model, Preprocessor, and SHAP Explainer loaded successfully.")
    
except Exception as e:
    print(f"❌ Initialization Error: Failed to load model or init SHAP: {e}")
    # Raise the error to prevent the server from starting with bad data
    raise

# ----------------
# 2. BUILD FLASK APP
# ----------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "MedBillingAI API is running.",
        "endpoints": ["/predict", "/explain"],
        "model_threshold": DECISION_THRESHOLD

    })

# -----------------
# 3. PREDICT ENDPOINT (FIXED)
# -----------------
@app.route("/predict", methods=["POST"])
def predict():
    """Predicts denial status using the custom DECISION_THRESHOLD."""
    try:
        data = request.json
        # CRITICAL: Ensure the DataFrame is created with the exact column order
        df = pd.DataFrame([data], columns=INPUT_FEATURE_ORDER) 

        # --- FIX: Pass the raw DataFrame (df) directly to the full Pipeline (model) ---
        # The Pipeline will automatically call preprocessor.transform(df) internally.
        y_proba = model.predict_proba(df)[0]
        # -----------------------------------------------------------------------------
        
        prob = y_proba[1] 

        # FIX: Apply the custom decision threshold
        # Denied (1) if probability is greater than or equal to the threshold
        pred = (prob >= DECISION_THRESHOLD).astype(int)

        return jsonify({
            "Denied": int(pred),
            "Denial_Probability": float(prob),
            "Threshold_Applied": DECISION_THRESHOLD
        })
    except Exception as e:
        # Provide a more informative error message to the client
        return jsonify({"error": f"Prediction failed: {str(e)}", "status": "error"}), 400

# -----------------
# 4. EXPLAIN ENDPOINT (Correct - requires manual preprocessing)
# -----------------
@app.route("/explain", methods=["POST"])
def explain():
    """Provides SHAP values for a single claim to explain the prediction."""
    try:
        data = request.json
        # CRITICAL FIX: Ensure the DataFrame is created with the exact column order
        df = pd.DataFrame([data], columns=INPUT_FEATURE_ORDER)

        # Since the explainer is initialized on the raw XGBoost model (not the pipeline), 
        # we MUST manually preprocess the input.
        processed = preprocessor.transform(df)

        # Compute SHAP values for Class 1 (Denial)
        shap_values = explainer.shap_values(processed)[0]
        base_value = explainer.expected_value
        
        # Ensure SHAP values and data are returned with feature names
        return jsonify({
            "shap_values": shap_values.tolist(),
            "feature_names": FEATURE_NAMES,
            "base_value": float(base_value),
            "message": "Higher SHAP value pushes stronger toward denial."
        })
    
    except Exception as e:
        # Corrected indentation for the except block
        return jsonify({"error": f"Explanation failed: {str(e)}", "status": "error"}), 400

# -----------------
# 5. RUN SERVER
# -----------------
if __name__ == "__main__":
    app.run(debug=True)