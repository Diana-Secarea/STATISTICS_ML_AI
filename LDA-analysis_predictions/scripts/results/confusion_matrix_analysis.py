import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
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
output_file_path = os.path.join(base_dir, 'confusion_matrix_analysis_output.txt')
output_file = open(output_file_path, 'w', encoding='utf-8')

# Redirect stdout to both console and file
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, output_file)

# Write header with timestamp
print(f"Confusion Matrix Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

# STEP 2: Generate Confusion Matrix
print("\n" + "="*60)
print("STEP 2: Generating Confusion Matrix")
print("="*60)

# Make predictions
y_pred = lda.predict(X)

# Calculate confusion matrix
cm = confusion_matrix(y, y_pred)

print(f"\nConfusion Matrix Shape: {cm.shape}")
print(f"Number of classes: {len(income_order)}")

# STEP 3: Display Confusion Matrix
print("\n" + "="*60)
print("STEP 3: Confusion Matrix (Raw Counts)")
print("="*60)

cm_df = pd.DataFrame(
    cm,
    index=[f'True {name}' for name in income_order],
    columns=[f'Predicted {name}' for name in income_order]
)

print("\nConfusion Matrix:")
print(cm_df.to_string())

# STEP 4: Normalized Confusion Matrix
print("\n" + "="*60)
print("STEP 4: Normalized Confusion Matrix (Percentages)")
print("="*60)

# Normalize by row (true labels)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_normalized_df = pd.DataFrame(
    cm_normalized * 100,  # Convert to percentages
    index=[f'True {name}' for name in income_order],
    columns=[f'Predicted {name}' for name in income_order]
)

print("\nNormalized Confusion Matrix (%):")
print(cm_normalized_df.round(2).to_string())

# STEP 5: Calculate Accuracy Metrics
print("\n" + "="*60)
print("STEP 5: Accuracy Metrics")
print("="*60)

# Overall accuracy
overall_accuracy = np.trace(cm) / np.sum(cm)
print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
print(f"Correct predictions: {np.trace(cm)} out of {np.sum(cm)}")

# Per-class accuracy (diagonal elements)
print("\nPer-Class Accuracy:")
for i, group in enumerate(income_order):
    if cm[i, i] > 0:
        class_accuracy = cm[i, i] / cm[i, :].sum()
        print(f"  {group}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%) - {cm[i, i]}/{cm[i, :].sum()} correct")

# STEP 6: Identify Misclassifications
print("\n" + "="*60)
print("STEP 6: Misclassification Analysis")
print("="*60)

print("\nMisclassification Patterns:")

misclassifications = []
for i, true_group in enumerate(income_order):
    for j, pred_group in enumerate(income_order):
        if i != j and cm[i, j] > 0:
            misclassifications.append({
                'True Group': true_group,
                'Predicted Group': pred_group,
                'Count': cm[i, j],
                'Percentage': (cm[i, j] / cm[i, :].sum() * 100)
            })

if misclassifications:
    misclass_df = pd.DataFrame(misclassifications)
    misclass_df = misclass_df.sort_values('Count', ascending=False)
    print(misclass_df.to_string(index=False))
else:
    print("  No misclassifications found (perfect classification)")

# STEP 7: Visualize Confusion Matrix
print("\n" + "="*60)
print("STEP 7: Creating Confusion Matrix Visualizations")
print("="*60)

# Create plots directory if it doesn't exist
plots_dir = os.path.join(base_dir, 'confusion_matrix_plots')
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: Raw confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
            linewidths=0.5, linecolor='gray', square=True)
plt.title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('True Income Group', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Income Group', fontsize=12, fontweight='bold')
plt.tight_layout()
raw_plot_path = os.path.join(plots_dir, 'confusion_matrix_raw.png')
plt.savefig(raw_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: {raw_plot_path}")

# Plot 2: Normalized confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized_df, annot=True, fmt='.1f', cmap='YlOrRd', 
            cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5, 
            linecolor='gray', square=True, vmin=0, vmax=100)
plt.title('Normalized Confusion Matrix (%)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('True Income Group', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Income Group', fontsize=12, fontweight='bold')
plt.tight_layout()
norm_plot_path = os.path.join(plots_dir, 'confusion_matrix_normalized.png')
plt.savefig(norm_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  [OK] Saved: {norm_plot_path}")

print(f"\nAll plots saved to: {plots_dir}")

# STEP 8: Interpretation
print("\n" + "="*60)
print("STEP 8: Interpretation")
print("="*60)

print("\n" + "-"*60)
print("Key Findings:")
print("-"*60)

# Check diagonal (correct predictions)
diagonal_sum = np.trace(cm)
total = np.sum(cm)
print(f"\n1. Overall Performance:")
print(f"   - {diagonal_sum} out of {total} predictions were correct ({overall_accuracy*100:.2f}%)")

# Check for low income group issues
low_income_idx = 0  # Assuming Low income is class 0
low_income_total = cm[low_income_idx, :].sum()
low_income_correct = cm[low_income_idx, low_income_idx]

print(f"\n2. Low Income Group Classification:")
print(f"   - Total samples: {low_income_total}")
print(f"   - Correctly classified: {low_income_correct}")
if low_income_total > 0:
    low_income_accuracy = low_income_correct / low_income_total
    print(f"   - Accuracy: {low_income_accuracy:.4f} ({low_income_accuracy*100:.2f}%)")

# Find most common misclassifications
if misclassifications:
    most_common = misclass_df.iloc[0]
    print(f"\n3. Most Common Misclassification:")
    print(f"   - {most_common['True Group']} countries are often predicted as {most_common['Predicted Group']}")
    print(f"   - Occurs {most_common['Count']} times ({most_common['Percentage']:.1f}% of {most_common['True Group']} cases)")

# Check if any groups are perfectly classified
perfect_groups = []
for i, group in enumerate(income_order):
    if cm[i, i] == cm[i, :].sum() and cm[i, i] > 0:
        perfect_groups.append(group)

if perfect_groups:
    print(f"\n4. Perfect Classification:")
    print(f"   - The following groups were classified with 100% accuracy:")
    for group in perfect_groups:
        print(f"     * {group}")


# Save confusion matrix to CSV
cm_csv_path = os.path.join(base_dir, 'confusion_matrix.csv')
cm_df.to_csv(cm_csv_path)
print(f"\nConfusion matrix saved to: {cm_csv_path}")

# Save normalized confusion matrix to CSV
cm_norm_csv_path = os.path.join(base_dir, 'confusion_matrix_normalized.csv')
cm_normalized_df.to_csv(cm_norm_csv_path)
print(f"Normalized confusion matrix saved to: {cm_norm_csv_path}")

# Restore stdout and close file
sys.stdout = original_stdout
output_file.close()
print(f"\nOutput saved to: {output_file_path}")
