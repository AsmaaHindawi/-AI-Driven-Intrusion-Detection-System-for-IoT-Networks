import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Load Your Final Processed Dataset with Selected Features and Label ===
selected_dataset_path = r"C:\Users\Youssef Hindawi\Desktop\IOT\CICIoT2023_selected_features.csv"

if not os.path.exists(selected_dataset_path):
    raise FileNotFoundError(f"Dataset not found at: {selected_dataset_path}")

df = pd.read_csv(selected_dataset_path)

# === Compute Correlation Matrix (including label column) ===
correlation_matrix = df.corr(method='pearson')

# === Plot the Correlation Heatmap with Highlighted Label Row ===
plt.figure(figsize=(18, 10))
sns.set(style="white")

# Generate the heatmap
heatmap = sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap='coolwarm',
    center=0,
    cbar_kws={'label': 'Pearson Correlation'},
    linewidths=0.5,
    linecolor='gray'
)

# Highlight the label row and column
label_index = list(correlation_matrix.columns).index('label')
for i in range(len(correlation_matrix)):
    heatmap.add_patch(plt.Rectangle((label_index, i), 1, 1, fill=False, edgecolor='gold', lw=2))
    heatmap.add_patch(plt.Rectangle((i, label_index), 1, 1, fill=False, edgecolor='gold', lw=2))

# Titles and Labels
plt.title("Feature Correlation Heatmap (with Highlight on Label)", fontsize=14, pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Show Plot
plt.show()
