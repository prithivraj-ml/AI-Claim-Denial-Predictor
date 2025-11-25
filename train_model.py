import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, make_scorer, f1_score, precision_score, recall_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import uniform, randint
import xgboost as xgb
import joblib


# We define a custom scorer to optimize for the F1-Score of the minority class (Denied=1).
f1_scorer = make_scorer(f1_score, pos_label=1)


#
#1. LOAD DATA
#
df = pd.read_csv("claims_data.csv")
#
#2. FEATURE SELECTION
#

#Adding this because of data leakage
df = df.drop('Denial_Reason', axis=1)
#

target = "Denied"
y = df [target]
X = df.drop(columns=[target])
# Identify categorical & numeric columns
categorical_cols = X.select_dtypes (include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes (include=["int64", "float64"]).columns.tolist()
#3. PREPROCESSING PIPELINE
preprocessor = ColumnTransformer (transformers=[("cat", OneHotEncoder (handle_unknown="ignore", sparse_output=False),categorical_cols),("num",StandardScaler(), numeric_cols)])
#
#4. XGBOOST MODEL
#
# The distributions define the range of values the search will randomly sample.
param_dist = {
    'n_estimators': randint(50, 500),         # Number of trees
    'max_depth': randint(3, 10),              # Maximum depth of a tree
    'learning_rate': uniform(0.01, 0.3),      # Step size shrinkage
    'subsample': uniform(0.6, 0.4),           # Subsample ratio of the training instances
    'colsample_bytree': uniform(0.6, 0.4),    # Subsample ratio of columns when constructing each tree
    'gamma': uniform(0, 0.5)                  # Minimum loss reduction required to make a further partition
}


#Adding ratio for using the weight
# Find the ratio of negative to positive samples (4108 / 892 = ~4.6)
ratio = len(df[df['Denied'] == 0]) / len(df[df['Denied'] == 1])

base_xgb = xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42, use_label_encoder=False, eval_metric='logloss')

#FuLL pipeline
search_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", base_xgb)])
#5. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#6. DEFINE SEARCH OBJECT
#
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    estimator=search_pipeline, # Use the pipeline here
    param_distributions={'model__' + k: v for k, v in param_dist.items()}, # Prepends 'model__' to param names
    n_iter=100, scoring=f1_scorer, cv=kfold, verbose=2, random_state=42, n_jobs=-1)
#
#7. FIT SEARCH
#
random_search.fit(X_train, y_train) # Pass the raw X_train, the pipeline handles preprocessing
best_clf = random_search.best_estimator_ # The best *pipeline*
# ----------------------------------------------------------------------
# ADDED NEW THRESHOLD TUNING BLOCK 
# ----------------------------------------------------------------------

# 1. Get raw prediction probabilities
y_proba = best_clf.predict_proba(X_test)
y_scores = y_proba[:, 1] # Probability of the positive class (Class 1: Denied)

# 2. Calculate Precision, Recall, and Thresholds
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

# 3. Find the threshold that maximizes the F1-Score (a balanced choice)
best_threshold = 0.5
best_f1 = 0

for p, r, t in zip(precisions, recalls, thresholds):
    # Avoid division by zero if both p and r are zero (shouldn't happen here, but safe practice)
    if (p + r) > 0:
        f1 = 2 * (p * r) / (p + r)
    else:
        continue
    
    # Keep track of the best F1-score and its threshold
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

# 4. Apply the new optimal threshold to get final predictions
y_pred_tuned = (y_scores >= best_threshold).astype(int)

#8. EVALUATE AND SAVE
print("\n=======================================================")
print(" OPTIMIZED MODEL PERFORMANCE (with Threshold Tuning)")
print("=======================================================\n")
print(f"Optimal Threshold for Max F1-Score: {best_threshold:.4f}")
print("Accuracy:", accuracy_score(y_test, y_pred_tuned)) # Use tuned predictions
print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned)) # Use tuned predictions
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_tuned)) # Use tuned predictions

# SAVE MODEL
joblib.dump(best_clf, "model_xgb_optimized.pkl")
print("\nModel saved as model_xgb_optimized.pkl")
#Save encoder & scaler separately (optional)
joblib.dump(preprocessor, "encoder_scaler.pkl")
print("Preprocessor saved as encoder_scaler.pkl")
print("\nTraining complete.")