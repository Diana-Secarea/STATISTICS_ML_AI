import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import sys
from datetime import datetime

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
output_file_path = os.path.join(base_dir, 'discriminant_coefficients_output.txt')
output_file = open(output_file_path, 'w', encoding='utf-8')

# Redirect stdout to both console and file
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, output_file)

# Write header with timestamp
print(f"Discriminant Coefficients Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

# STEP 2: Extract Discriminant Coefficients
print("\n" + "="*60)
print("STEP 2: Discriminant Coefficients")
print("="*60)

# Create DataFrame with coefficients
# lda.coef_ shape: (n_classes - 1, n_features)
# For 4 classes, we have 3 discriminant functions
coefficients = pd.DataFrame(
    lda.coef_,
    columns=indicator_cols,
    index=[f'Discriminant Function {i+1}' for i in range(lda.coef_.shape[0])]
)

# Rename columns for better readability
coefficients.columns = ['GDP per capita', 'Unemployment rate', 'Inflation (CPI)', 
                        'Life expectancy', 'School enrollment']

print("\nDiscriminant Coefficients:")
print(coefficients.round(4).to_string())

# STEP 3: Variable Importance (Absolute Coefficients)
print("\n" + "="*60)
print("STEP 3: Variable Importance Analysis")
print("="*60)

# Calculate mean absolute coefficient across all discriminant functions
importance = coefficients.abs().mean(axis=0).sort_values(ascending=False)
importance_df = pd.DataFrame({
    'Feature': importance.index,
    'Mean Absolute Coefficient': importance.values
})

print("\nFeature Importance (Mean Absolute Coefficient):")
print(importance_df.round(4).to_string(index=False))

# STEP 4: Visualize Coefficients
print("\n" + "="*60)
print("STEP 4: Creating Coefficient Visualizations")
print("="*60)

# Create plots directory if it doesn't exist
plots_dir = os.path.join(base_dir, 'results_plots')
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: Bar chart of absolute coefficients for each discriminant function
fig, axes = plt.subplots(1, lda.coef_.shape[0], figsize=(6*lda.coef_.shape[0], 6))
if lda.coef_.shape[0] == 1:
    axes = [axes]

for idx, ax in enumerate(axes):
    abs_coef = coefficients.iloc[idx].abs().sort_values(ascending=True)
    ax.barh(abs_coef.index, abs_coef.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Absolute Coefficient Value', fontsize=10)
    ax.set_title(f'Discriminant Function {idx+1}\nVariable Importance', 
                 fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, abs_coef.max() * 1.1)

plt.tight_layout()
coef_plot_path = os.path.join(plots_dir, 'discriminant_coefficients.png')
plt.savefig(coef_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: {coef_plot_path}")

# Plot 2: Heatmap of coefficients
plt.figure(figsize=(10, 6))
sns.heatmap(coefficients.T, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Coefficient Value'}, linewidths=0.5)
plt.title('Discriminant Coefficients Heatmap', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Discriminant Function', fontsize=11)
plt.ylabel('Feature', fontsize=11)
plt.tight_layout()
heatmap_path = os.path.join(plots_dir, 'coefficients_heatmap.png')
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: {heatmap_path}")

# Plot 3: Mean absolute coefficients bar chart
plt.figure(figsize=(10, 6))
importance_sorted = importance.sort_values(ascending=True)
plt.barh(importance_sorted.index, importance_sorted.values, color='coral', alpha=0.7)
plt.xlabel('Mean Absolute Coefficient', fontsize=11)
plt.title('Overall Feature Importance\n(Mean Absolute Coefficient Across All Discriminant Functions)', 
          fontsize=12, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
importance_plot_path = os.path.join(plots_dir, 'feature_importance.png')
plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: {importance_plot_path}")

print(f"\nAll plots saved to: {plots_dir}")

# STEP 5: Classification Performance
print("\n" + "="*60)
print("STEP 5: Classification Performance")
print("="*60)

# Make predictions
y_pred = lda.predict(X)

# Confusion Matrix
print("\n" + "-"*60)
print("Confusion Matrix")
print("-"*60)
cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(cm, 
                     index=[f'True {name}' for name in income_order],
                     columns=[f'Pred {name}' for name in income_order])
print(cm_df.to_string())

# Calculate accuracy
accuracy = np.trace(cm) / np.sum(cm)
print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification Report
print("\n" + "-"*60)
print("Classification Report")
print("-"*60)
report = classification_report(y, y_pred, target_names=income_order, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(classification_report(y, y_pred, target_names=income_order))

# Save detailed report
report_df_path = os.path.join(base_dir, 'classification_report.csv')
report_df.to_csv(report_df_path)
print(f"\nDetailed classification report saved to: {report_df_path}")

# STEP 6: Interpretation
print("\n" + "="*60)
print("STEP 6: Interpretation")
print("="*60)

print("\nMost Important Features (by mean absolute coefficient):")
for i, (feature, importance_val) in enumerate(importance.items(), 1):
    print(f"  {i}. {feature}: {importance_val:.4f}")

print("\n" + "-"*60)
print("Classification Performance by Group:")
print("-"*60)
for i, group in enumerate(income_order):
    precision = report[group]['precision']
    recall = report[group]['recall']
    f1 = report[group]['f1-score']
    support = report[group]['support']
    print(f"\n{group}:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  Support: {support}")


# Restore stdout and close file
sys.stdout = original_stdout
output_file.close()
print(f"\nOutput saved to: {output_file_path}")
