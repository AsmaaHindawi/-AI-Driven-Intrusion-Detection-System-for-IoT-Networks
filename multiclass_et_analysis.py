# Import Libraries 
import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

# Suppress warnings
warnings.filterwarnings('ignore')

# Display all columns
pd.set_option('display.max_columns', None)

# === Load Dataset from Multiple CSV Files ===
dataset_folder = r"C:\Users\Youssef Hindawi\Desktop\IOT\CICIoT2023"

# Get list of all CSV files in the folder
csv_files = [os.path.join(dataset_folder, file) for file in os.listdir(dataset_folder) if file.endswith('.csv')]

# Read and concatenate all CSV files into a single DataFrame
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Print dataset column names for verification
print("Columns in dataset:", df.columns.tolist())

# Ensure 'label' column exists
if 'label' not in df.columns:
    raise ValueError("ERROR: 'label' column not found in the dataset! Please check column names.")

# Drop unnecessary columns
df.drop(columns=['Binary Class'], inplace=True, errors='ignore')

# Encode the 'label' column
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Print initial class distribution
print("\nClass Distribution:\n", df['label'].value_counts())

# **Filter out classes with fewer than 2 samples**
df = df[df.groupby('label')['label'].transform('count') >= 2]

# **Check again to confirm all classes have at least 2 samples**
print("\nUpdated Class Distribution After Removing Small Classes:\n", df['label'].value_counts())

# Define feature matrix and target variable
X = df.drop(columns=['label'])
y = df['label']

# **Check again if the smallest class has at least 2 samples**
if y.value_counts().min() < 2:
    raise ValueError("ERROR: Some classes still have fewer than 2 samples after filtering!")

# Split data into train and temp sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# **Recheck classes in `y_temp` and remove those with fewer than 2 samples**
y_temp_counts = y_temp.value_counts()
valid_classes = y_temp_counts[y_temp_counts >= 2].index

# Keep only valid classes
X_temp = X_temp[y_temp.isin(valid_classes)]
y_temp = y_temp[y_temp.isin(valid_classes)]

# **Final check before splitting validation and test sets**
if y_temp.value_counts().min() < 2:
    raise ValueError("ERROR: Some classes in y_temp still have fewer than 2 samples, breaking stratification!")

# Split validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Print dataset sizes
print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Testing set size: {len(X_test)}")

# Feature selection using Extra Trees Classifier
et_model = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_model.fit(X_train, y_train)

# Get feature importance scores
feature_importances = pd.Series(et_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Select top 25 features
top_25_features = feature_importances.head(25).index.tolist()

print("\nTop 25 features from Extra Trees:")
print(top_25_features)

# Remove unselected features
unselected_features = X.columns.difference(top_25_features).tolist()
print("\nUnselected features:", unselected_features)

# Keep only selected features
X_train = X_train[top_25_features]
X_val = X_val[top_25_features]
X_test = X_test[top_25_features]

# Data visualization for Multiclass Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train)
plt.title("Multiclass Distribution in Training Set")
plt.xlabel("Attack Category (Encoded)")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Boxplots for selected top features
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(18, 20))

for idx, feature in enumerate(top_25_features):
    row, col = divmod(idx, 5)
    sns.boxplot(data=df, x='label', y=feature, ax=axes[row, col])
    axes[row, col].set_title(f'{feature} by Attack Category')
    axes[row, col].set_xlabel('Attack Category')
    axes[row, col].set_ylabel(feature)

plt.tight_layout()
plt.show()

# Feature correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df[top_25_features].corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Save cleaned dataset with selected features
selected_features_path = r"C:\Users\Youssef Hindawi\Desktop\IOT\CICIoT2023_selected_features.csv"
df_selected = df[top_25_features + ['label']]
df_selected.to_csv(selected_features_path, index=False)
print(f"\nDataset with selected features saved successfully at {selected_features_path}.")
