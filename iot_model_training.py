# Import required libraries
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from imblearn.combine import SMOTETomek

# ------------------- DIRECTORY SETUP -------------------
# Paths for dataset and output storage
DATASET_DIRECTORY = r"C:\Users\Youssef Hindawi\Desktop\IOT\CICIoT2023"
PREPROCESSED_OBJECTS_PATH = r"C:\Users\Youssef Hindawi\Desktop\IOT\preprocessed_objects"
MODEL_SAVE_PATH = r"C:\Users\Youssef Hindawi\Desktop\IOT\models"
BALANCED_DATA_PATH = r"C:\Users\Youssef Hindawi\Desktop\IOT\Data_Balanced"
METRICS_PATH = os.path.join(MODEL_SAVE_PATH, "evaluation_metrics.txt")

# Ensure directories exist
os.makedirs(PREPROCESSED_OBJECTS_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(BALANCED_DATA_PATH, exist_ok=True)

# ------------------- LOAD DATASET -------------------
# Load the dataset
dataset_file = os.path.join(DATASET_DIRECTORY, "Data160.csv")
if not os.path.exists(dataset_file):
    raise FileNotFoundError(f"File not found: {dataset_file}")

print(f"Loading file: {dataset_file}")
dataset = pd.read_csv(dataset_file)

# Display initial dataset structure
print("Dataset loaded successfully!")
print(dataset.head())
print("Initial dataset info:")
print(dataset.info())

# ------------------- MEMORY OPTIMIZATION -------------------
# Reduce memory usage by downcasting numerical columns
for col in dataset.select_dtypes(include=['float64', 'int64']).columns:
    dataset[col] = pd.to_numeric(dataset[col], downcast='float')

print("Optimized data types for memory efficiency.")

# ------------------- ATTACK TYPE MAPPING -------------------
# Map attack labels to general categories
attack_mapping = {
    'DDoS-RSTFINFlood': 'DDoS', 'DoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS',
    'DoS-UDP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS',
    'MITM-ArpSpoofing': 'Spoofing', 'DNS_Spoofing': 'Spoofing',
    'BruteForce': 'BruteForce'
}

dataset['AttackCategory'] = dataset['label'].map(attack_mapping)
dataset = dataset[dataset['AttackCategory'].notna()]

print("Filtered dataset for selected attack categories:")
print(dataset['AttackCategory'].value_counts())

# ------------------- LABEL ENCODING -------------------
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(dataset['AttackCategory'])

# ------------------- FEATURE SELECTION -------------------
selected_features = [
    'flow_duration', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
    'ack_flag_number', 'syn_count', 'fin_count', 'TCP', 'UDP', 'Tot sum', 'Weight'
]

# Ensure all selected features exist in dataset
selected_features = [feature for feature in selected_features if feature in dataset.columns]
X = dataset[selected_features]
y = labels_encoded

# ------------------- STRATIFIED TRAIN-TEST SPLIT -------------------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

# ------------------- SMOTE-TOMEK BALANCING -------------------
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

print("Balanced Class Distribution (After SMOTE-Tomek):")
print(pd.Series(y_train_balanced).value_counts())

# Save balanced dataset
balanced_data = pd.DataFrame(X_train_balanced, columns=selected_features)
balanced_data['AttackCategory'] = label_encoder.inverse_transform(y_train_balanced)
balanced_data.to_csv(os.path.join(BALANCED_DATA_PATH, "balanced_data.csv"), index=False)

print("Balanced dataset saved.")

# ------------------- FEATURE SCALING -------------------
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for future use
scaler_path = os.path.join(PREPROCESSED_OBJECTS_PATH, "robust_scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

print("Feature scaling completed, and scaler saved.") 

# ------------------- HYPERPARAMETER TUNING -------------------
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.03, 0.05, 0.07],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [3, 5, 7],
    'reg_lambda': [1.0, 1.5, 2.0],  # L2 Regularization
    'reg_alpha': [0.1, 0.3, 0.5]    # L1 Regularization
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), random_state=42),
    param_grid, scoring='f1_weighted', cv=3, verbose=2
)
grid_search.fit(X_train_scaled, y_train_balanced)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# ------------------- TRAIN FINAL MODEL -------------------
model = xgb.XGBClassifier(**best_params)
model.fit(X_train_scaled, y_train_balanced)

# Save the trained model
model_path = os.path.join(MODEL_SAVE_PATH, "xgboost_smote_tomek.json")
model.save_model(model_path)

print("Optimized XGBoost model trained and saved at:", model_path)

# ------------------- MODEL EVALUATION -------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# ------------------- SAVE METRICS -------------------
with open(METRICS_PATH, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ------------------- FEATURE IMPORTANCE PLOT -------------------
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': selected_features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_df['Importance'], y=feature_importance_df['Feature'], palette='viridis')
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
