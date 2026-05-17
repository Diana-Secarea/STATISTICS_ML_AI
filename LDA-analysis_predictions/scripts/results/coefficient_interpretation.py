import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Get the parent directory (project root)
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a class to write to both console and file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Open output file
output_file_path = os.path.join(base_dir, 'coefficient_interpretation_output.txt')
output_file = open(output_file_path, 'w', encoding='utf-8')

# Redirect stdout to both console and file
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, output_file)

# Write header with timestamp
print(f"Discriminant Coefficients Interpretation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
print()

# Load data and model
print("="*60)
print("STEP 1: Loading data and model")
print("="*60)

processed_path = os.path.join(base_dir, 'merged_dataset_processed.csv')
df = pd.read_csv(processed_path)
print(f"Dataset shape: {df.shape}")

# Define features
indicator_cols = ['GDP_per_capita', 'Unemployment_rate', 'Inflation_CPI', 
                  'Life_expectancy', 'School_enrollment']

X = df[indicator_cols].values
y = df['Income_group_encoded'].values

# Load or create LDA model
model_path = os.path.join(base_dir, 'lda_model.pkl')
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        lda = pickle.load(f)
    print(f"Loaded LDA model from: {model_path}")
else:
    # Fit new model
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    print("Fitted new LDA model")

# Class names
income_order = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']

# STEP 2: Extract and Display Coefficients
print("\n" + "="*60)
print("STEP 2: Discriminant Coefficients")
print("="*60)

# Create DataFrame with coefficients
coefficients = pd.DataFrame(
    lda.coef_,
    columns=indicator_cols,
    index=[f'Discriminant Function {i+1}' for i in range(lda.coef_.shape[0])]
)

# Rename columns for better readability
coefficients.columns = ['GDP per capita', 'Unemployment rate', 'Inflation (CPI)', 
                        'Life expectancy', 'School enrollment']

print("\nDiscriminant Coefficients Table:")
print(coefficients.round(4).to_string())

# STEP 3: Calculate Absolute Coefficients for Visualization
print("\n" + "="*60)
print("STEP 3: Absolute Coefficient Analysis")
print("="*60)

# Calculate mean absolute coefficient across all discriminant functions
abs_coefficients = coefficients.abs()
mean_abs_coef = abs_coefficients.mean(axis=0).sort_values(ascending=False)

print("\nMean Absolute Coefficients (Feature Importance):")
importance_df = pd.DataFrame({
    'Feature': mean_abs_coef.index,
    'Mean Absolute Coefficient': mean_abs_coef.values
})
print(importance_df.round(4).to_string(index=False))

# STEP 4: Create Bar Chart Visualization
print("\n" + "="*60)
print("STEP 4: Creating Bar Chart Visualization")
print("="*60)

# Create plots directory if it doesn't exist
plots_dir = os.path.join(base_dir, 'coefficient_plots')
os.makedirs(plots_dir, exist_ok=True)

# Bar chart of absolute coefficients for each discriminant function
fig, axes = plt.subplots(1, lda.coef_.shape[0], figsize=(7*lda.coef_.shape[0], 6))
if lda.coef_.shape[0] == 1:
    axes = [axes]

for idx, ax in enumerate(axes):
    abs_coef = abs_coefficients.iloc[idx].sort_values(ascending=True)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(abs_coef)))
    ax.barh(abs_coef.index, abs_coef.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Absolute Coefficient Value', fontsize=11, fontweight='bold')
    ax.set_title(f'Discriminant Function {idx+1}\nVariable Importance', 
                 fontsize=12, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax.set_xlim(0, abs_coef.max() * 1.15)
    
    # Add value labels on bars
    for i, (feature, value) in enumerate(zip(abs_coef.index, abs_coef.values)):
        ax.text(value + abs_coef.max() * 0.02, i, f'{value:.3f}', 
                va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
bar_plot_path = os.path.join(plots_dir, 'coefficient_bar_chart.png')
plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: {bar_plot_path}")

# Overall feature importance bar chart
plt.figure(figsize=(10, 6))
mean_abs_sorted = mean_abs_coef.sort_values(ascending=True)
colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(mean_abs_sorted)))
plt.barh(mean_abs_sorted.index, mean_abs_sorted.values, color=colors, alpha=0.8, 
         edgecolor='black', linewidth=0.5)
plt.xlabel('Mean Absolute Coefficient', fontsize=12, fontweight='bold')
plt.title('Overall Feature Importance\n(Mean Absolute Coefficient Across All Discriminant Functions)', 
          fontsize=13, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='x', linestyle='--')

# Add value labels
for i, (feature, value) in enumerate(zip(mean_abs_sorted.index, mean_abs_sorted.values)):
    plt.text(value + mean_abs_sorted.max() * 0.02, i, f'{value:.3f}', 
             va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
importance_plot_path = os.path.join(plots_dir, 'overall_feature_importance.png')
plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: {importance_plot_path}")

print(f"\nAll plots saved to: {plots_dir}")

# STEP 5: Feature Importance Ranking
print("\n" + "="*60)
print("STEP 5: Feature Importance Ranking")
print("="*60)

print("\nFeature Importance Ranking:")
for i, (feature, importance_val) in enumerate(mean_abs_coef.items(), 1):
    print(f"{i}. {feature}: {importance_val:.4f}")

# Restore stdout and close file
sys.stdout = original_stdout
output_file.close()
print(f"\nOutput saved to: {output_file_path}")
