
# 🌟 **MedBillingAI Claim Denial Predictor**

This repository contains the full pipeline for training and deploying an XGBoost-based machine learning model to predict healthcare claim denial risk. The goal is to maximize the identification of future denied claims (high Recall for the minority class) to enable proactive intervention before payment is rejected.

---

# 📁 **Project Structure**

| File                                  | Description                                                                                                                                                                                                                   |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **medbillingai_dataset_generator.py** | Python script used to create synthetic, yet realistic, healthcare claims data (claims_data.csv).                                                                                                                              |
| **claims_data.csv**                   | The synthetic dataset used for training, containing claim features and the Denied target variable.                                                                                                                            |
| **train_model.py**                    | Core Training Script. Implements data preprocessing, stratified train/test split, XGBoost modeling with class weighting, hyperparameter tuning via RandomizedSearchCV, and custom F1-score optimization via threshold tuning. |
| **model_xgb_optimized.pkl**           | The serialized final trained machine learning pipeline (includes the preprocessor and the best XGBoost model).                                                                                                                |
| **app.py**                            | Flask API. Serves the trained model via two endpoints (/predict and /explain), handling preprocessing and using the optimal decision threshold for live predictions.                                                          |
| **dashboard.py**                      | Streamlit Dashboard. A user interface for submitting new claims to the Flask API and visualizing the prediction and the SHAP-based explanation.                                                                               |

---

# 🔧 **Prerequisites**

To run this project, you need Python and the following libraries:

```
pip install pandas numpy scikit-learn xgboost flask requests joblib shap streamlit
```

---

# 🧪 **1. Data Generation**

The data is synthetic but mimics real-world healthcare claim structures.

### Run the Generator:

This script creates the input file claims_data.csv.

```
python medbillingai_dataset_generator.py
```

### Key Features:

The dataset includes common claim features such as:
Patient_Age, Gender, ICD10_Code, CPT_Code, Provider_Type, Claim_Amount, Processing_Days, and the target variable Denied (0=Approved, 1=Denied).

---

# 🤖 **2. Model Training and Optimization**

The train_model.py script executes a comprehensive machine learning workflow designed to handle the class imbalance inherent in denial prediction (where denied claims are the minority class).

---

## ⚙️ **Step 2.1: Data Preprocessing Pipeline**

A ColumnTransformer is used to apply different transformations to different feature groups:

* **Categorical Features:** Processed using OneHotEncoder to convert non-numeric text features (like ICD10_Code, Provider_Type) into binary columns.
* **Numeric Features:** Imputed (filling missing values with the median) and then Scaled using StandardScaler to normalize the values.

---

## ⚖️ **Step 2.2: Handling Class Imbalance**

Denial prediction is imbalanced (far more approved claims than denied claims). To address this, two methods were applied:

* **XGBoost scale_pos_weight:** Weighted by Count(0) / Count(1)
* **Stratified K-Fold:** Maintains class balance across folds.

---

## 🔍 **Step 2.3: Hyperparameter Search**

RandomizedSearchCV is used to efficiently search for the best model parameters (like n_estimators, max_depth, learning_rate) that maximize the F1-Score for the Denied (Class 1).

---

## 🎯 **Step 2.4: F1-Score Threshold Tuning (Critical Step)**

Since the raw XGBoost output is a probability, the standard 0.5 threshold may not be optimal for a skewed dataset. This step finds the specific probability threshold that maximizes the F1-Score on the test set by analyzing the Precision-Recall curve.

<img width="1460" height="842" alt="image" src="https://github.com/user-attachments/assets/b6bc1518-aa6e-4724-9542-5267383d422e" />



---

## ▶️ **Execution**

Run the training script:

```
python train_model.py
```

### Output Files:

* **model_xgb_optimized.pkl:** The complete Scikit-learn Pipeline (Preprocessor + XGBoost model).

---

# 🌐 **3. API Deployment (Flask)**

The app.py file creates a REST API to serve predictions.

* **Load Model:** Loads the model_xgb_optimized.pkl and initializes the SHAP explainer.
* **Prediction (/predict):** Takes a JSON claim object, applies preprocessing, generates denial probability, applies decision threshold, returns Denied (0 or 1).
* **Explanation (/explain):** Returns SHAP values explaining prediction.

### Execution

```
python app.py
```

The API will be available at:

```
http://127.0.0.1:5000
```

---

# 🖥️ **4. User Interface (Streamlit Dashboard)**

The dashboard.py provides a clean, interactive front-end for testing the API.

* **Claim Submission:** Users input claim features via form.
* **Prediction:** Sent to Flask /predict endpoint.
* **Visualization:** Sends data to /explain to generate SHAP Waterfall Plot.

<img width="1879" height="909" alt="Screenshot 2025-11-25 152900" src="https://github.com/user-attachments/assets/bff7d7db-1a19-41b5-9ef6-5d11b6bbc2ec" />



### Execution

```
streamlit run dashboard.py
```

The dashboard will open in your browser, allowing you to interact with the deployed AI model.

