import pandas as pd
import os
import numpy as np
from sklearn.metrics import classification_report
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
output_file_path = os.path.join(base_dir, 'classification_report_output.txt')
output_file = open(output_file_path, 'w', encoding='utf-8')

# Redirect stdout to both console and file
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, output_file)

# Write header with timestamp
print(f"LDA Classification Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

# STEP 2: Make Predictions
print("\n" + "="*60)
print("STEP 2: Making Predictions")
print("="*60)

y_pred = lda.predict(X)
print(f"Predictions completed for {len(y_pred)} samples")

# STEP 3: Classification Report
print("\n" + "="*60)
print("STEP 3: Classification Report")
print("="*60)

# Generate classification report
report = classification_report(y, y_pred, target_names=income_order, output_dict=True)

print("\nDetailed Classification Report:")
print("="*60)
print(classification_report(y, y_pred, target_names=income_order))

# STEP 4: Detailed Metrics by Class
print("\n" + "="*60)
print("STEP 4: Performance Metrics by Income Group")
print("="*60)

# Create DataFrame for better visualization
metrics_data = []
for group in income_order:
    if group in report:
        metrics_data.append({
            'Income Group': group,
            'Precision': report[group]['precision'],
            'Recall': report[group]['recall'],
            'F1-Score': report[group]['f1-score'],
            'Support': int(report[group]['support'])
        })

metrics_df = pd.DataFrame(metrics_data)
print("\nPerformance Metrics Table:")
print(metrics_df.round(3).to_string(index=False))

# STEP 5: Overall Performance
print("\n" + "="*60)
print("STEP 5: Overall Model Performance")
print("="*60)

print(f"\nOverall Accuracy: {report['accuracy']:.4f} ({report['accuracy']*100:.2f}%)")
print(f"Macro Average Precision: {report['macro avg']['precision']:.4f}")
print(f"Macro Average Recall: {report['macro avg']['recall']:.4f}")
print(f"Macro Average F1-Score: {report['macro avg']['f1-score']:.4f}")
print(f"Weighted Average Precision: {report['weighted avg']['precision']:.4f}")
print(f"Weighted Average Recall: {report['weighted avg']['recall']:.4f}")
print(f"Weighted Average F1-Score: {report['weighted avg']['f1-score']:.4f}")

# STEP 6: Interpretation
print("\n" + "="*60)
print("STEP 6: Interpretation")
print("="*60)

print("\n" + "-"*60)
print("Performance by Income Group:")
print("-"*60)

for group in income_order:
    if group in report:
        precision = report[group]['precision']
        recall = report[group]['recall']
        f1 = report[group]['f1-score']
        support = int(report[group]['support'])
        
        print(f"\n{group}:")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        print(f"  Support: {support}")

print("\n" + "-"*60)
print("Key Insights:")
print("-"*60)

# Find best and worst performing groups
f1_scores = {group: report[group]['f1-score'] for group in income_order if group in report}
best_group = max(f1_scores, key=f1_scores.get)
worst_group = min(f1_scores, key=f1_scores.get)

print(f"\nBest performing group: {best_group} (F1-Score: {f1_scores[best_group]:.3f})")
print(f"Worst performing group: {worst_group} (F1-Score: {f1_scores[worst_group]:.3f})")


# Save detailed report to CSV
report_df = pd.DataFrame(report).transpose()
report_csv_path = os.path.join(base_dir, 'detailed_classification_report.csv')
report_df.to_csv(report_csv_path)
print(f"\nDetailed classification report saved to: {report_csv_path}")

# Restore stdout and close file
sys.stdout = original_stdout
output_file.close()
print(f"\nOutput saved to: {output_file_path}")
