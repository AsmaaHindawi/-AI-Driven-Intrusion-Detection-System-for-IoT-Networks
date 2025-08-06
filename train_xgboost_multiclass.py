# Import Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

# === Load Dataset ===
data_path = r"C:\Users\Youssef Hindawi\Desktop\IOT\CICIoT2023_selected_features.csv"

# Verify if the dataset file exists
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Error: The dataset file '{data_path}' was not found. Please check the file path.")

df = pd.read_csv(data_path)

# === Display Available Columns ===
print("Columns in dataset:", df.columns.tolist())

# === Identify Label Column ===
if "Multiclass" in df.columns:
    label_column = "Multiclass"
elif "label" in df.columns:
    label_column = "label"
else:
    raise KeyError("Error: No valid label column found! Check the dataset structure.")

print(f"\nUsing label column: {label_column}")

# === Remove Classes with Fewer Than 10 Samples ===
min_samples_threshold = 10
class_counts = df[label_column].value_counts()
valid_classes = class_counts[class_counts >= min_samples_threshold].index
df = df[df[label_column].isin(valid_classes)]  

# === Verify Class Distribution After Filtering ===
print("\nUpdated Class Distribution After Removing Small Classes:\n", df[label_column].value_counts())

# === Feature Selection ===
feature_columns = [col for col in df.columns if col not in ["label", "Multiclass"]]
X = df[feature_columns]
y = df[label_column]

# === Encode Labels ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# === Feature Scaling ===
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# === Splitting Data ===
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# === Ensure No Stratify Error ===
unique_classes_y_temp = np.unique(y_temp)
if len(unique_classes_y_temp) > 1:
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
else:
    print("\nWarning: Only one class in y_temp, performing random split instead!")
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

print(f"\nTraining set size: {len(y_train)}")
print(f"Validation set size: {len(y_val)}")
print(f"Testing set size: {len(y_test)}")

# === Hyperparameter Tuning with GridSearchCV ===
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [6, 8, 10],
    "n_estimators": [100, 200, 300],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
}

xgb_model = XGBClassifier(
    objective="multi:softmax",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring="accuracy",
    cv=3,  # 3-fold cross-validation
    verbose=2,
    n_jobs=-1
)

print("\nStarting hyperparameter tuning with GridSearchCV...")
grid_search.fit(X_train, y_train)

# === Best Hyperparameters ===
best_params = grid_search.best_params_
print("\nBest Hyperparameters:", best_params)

# === Train Final Model with Best Parameters ===
best_xgb = XGBClassifier(
    objective="multi:softmax",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42,
    **best_params  # Use best params from GridSearchCV
)

best_xgb.fit(X_train, y_train)

# === Model Evaluation ===
y_pred = best_xgb.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n=== Model Performance ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_.astype(str)))

# === Save Best Model & Scaler ===
model_path = r"C:\Users\Youssef Hindawi\Desktop\IOT\xgb_multiclass_best_model.pkl"
scaler_path = r"C:\Users\Youssef Hindawi\Desktop\IOT\robust_scaler.pkl"
encoder_path = r"C:\Users\Youssef Hindawi\Desktop\IOT\label_encoder.pkl"

joblib.dump(best_xgb, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoder, encoder_path)

print("\nBest Model, Scaler, and Label Encoder saved successfully!")
