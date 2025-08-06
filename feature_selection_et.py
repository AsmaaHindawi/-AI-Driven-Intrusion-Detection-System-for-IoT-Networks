# Import Libraries 
import pandas as pd
import numpy as np
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier

# Suppress warnings
warnings.filterwarnings('ignore')

# === Load Dataset from Multiple CSV Files ===
data_folder = "C:\\Users\\Youssef Hindawi\\Desktop\\IOT\\CICIoT2023"

# Get list of all CSV files in the folder
csv_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.csv')]

# Read and concatenate all CSV files into a single DataFrame
df_list = [pd.read_csv(file) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

# Display column names before proceeding
print("Columns in dataset:", df.columns.tolist())

# Rename 'label' column to 'Multiclass' if it exists
if 'label' in df.columns:
    df.rename(columns={'label': 'Multiclass'}, inplace=True)

# Ensure 'Multiclass' exists before encoding
if 'Multiclass' not in df.columns:
    raise KeyError("Error: 'Multiclass' column not found in the dataset. Check column names!")

# Drop unnecessary columns if they exist
columns_to_drop = ['Binary Class']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Display dataset shape after dropping columns
print("Dataset shape after dropping unnecessary columns:", df.shape)

# Encoding Multiclass Labels
encoder = LabelEncoder()
df['Multiclass'] = encoder.fit_transform(df['Multiclass'])

# Define Features and Target
X = df.drop(columns=['Multiclass'])
y = df['Multiclass']

# Feature Scaling using RobustScaler
scaler = RobustScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split Data into Training, Testing, and Validation Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

# Display dataset sizes
print("Training set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Testing set size:", X_test.shape[0])

# Initialize Extra Trees Classifier (ET)
et = ExtraTreesClassifier()
et.fit(X, y)

# Get Feature Importances from Extra Trees
et_importances = et.feature_importances_

# Sort features by importance for Extra Trees
feature_names = df.drop(columns=['Multiclass']).columns
sorted_indices_et = et_importances.argsort()[::-1]
top_25_indices_et = sorted_indices_et[:25]

# Get top 25 feature names
top_25_features_et = [feature_names[i] for i in top_25_indices_et]

# Print the top 25 feature names from Extra Trees
print("\nTop 25 features from Extra Trees:")
for feature in top_25_features_et:
    print(feature)

# Create a DataFrame combining feature names and scores
score_et = pd.DataFrame({
    'Feature': feature_names,
    'ET Importance': et_importances
})

# Sort the DataFrame by importance (descending order)
score_et = score_et.sort_values(by='ET Importance', ascending=False)

# Get the top 25 features
top_25_features_et_df = score_et.head(25)

# Plot Feature Importance for Extra Trees
sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))
sns.barplot(x='ET Importance', y='Feature', data=top_25_features_et_df, palette='viridis')

# Set labels and title
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.title('Top 25 Feature Importance Ranking using Extra Trees')

# Show plot
plt.show()

# Get remaining unselected features
unselected_et = score_et[~score_et['Feature'].isin(top_25_features_et_df['Feature'])]
unselected_et_array = unselected_et['Feature'].values

# Display unselected features
print("\nUnselected features from Extra Trees:\n", unselected_et_array)
