import pandas as pd
import os
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
output_file_path = os.path.join(base_dir, 'lda_fitting_output.txt')
output_file = open(output_file_path, 'w', encoding='utf-8')

# Redirect stdout to both console and file
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, output_file)

# Write header with timestamp
print(f"LDA Model Fitting - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
print()

# Load the processed dataset
print("="*60)
print("STEP 1: Loading processed dataset")
print("="*60)
processed_path = os.path.join(base_dir, 'merged_dataset_processed.csv')
df = pd.read_csv(processed_path)
print(f"Dataset shape: {df.shape}")

# Define features (standardized indicator columns)
indicator_cols = ['GDP_per_capita', 'Unemployment_rate', 'Inflation_CPI', 
                  'Life_expectancy', 'School_enrollment']

# Extract features and target
X = df[indicator_cols].values
y = df['Income_group_encoded'].values

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls, count in zip(unique, counts):
    income_group = df[df['Income_group_encoded'] == cls]['Income group'].iloc[0]
    print(f"  Class {cls} ({income_group}): {count} samples")

# STEP 2: Fit LDA Model
print("\n" + "="*60)
print("STEP 2: Fitting LDA Model")
print("="*60)

# Initialize LDA with 2 components for 2D visualization
lda = LinearDiscriminantAnalysis(n_components=2)

# Fit and transform
print("Fitting LDA model...")
lda.fit(X, y)
X_lda = lda.transform(X)

print(f"\nLDA transformation complete!")
print(f"Original feature space: {X.shape}")
print(f"LDA-transformed space: {X_lda.shape}")

# STEP 3: Model Information
print("\n" + "="*60)
print("STEP 3: Model Information")
print("="*60)

print(f"\nExplained variance ratio:")
explained_variance = lda.explained_variance_ratio_
for i, var in enumerate(explained_variance):
    print(f"  LD{i+1}: {var:.4f} ({var*100:.2f}%)")

print(f"\nTotal explained variance: {explained_variance.sum():.4f} ({explained_variance.sum()*100:.2f}%)")

# Get class names for reference
income_order = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
class_names = [income_order[i] for i in range(len(income_order))]

print(f"\nClass names:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

# Save the transformed data
print("\n" + "="*60)
print("STEP 4: Saving Results")
print("="*60)

# Create DataFrame with LDA results
lda_results = df[['Country Name', 'Country Code', 'Income group', 'Income_group_encoded']].copy()
lda_results['LD1'] = X_lda[:, 0]
lda_results['LD2'] = X_lda[:, 1]

# Save to CSV
lda_results_path = os.path.join(base_dir, 'lda_transformed_data.csv')
lda_results.to_csv(lda_results_path, index=False)
print(f"LDA transformed data saved to: {lda_results_path}")

# Save the model components for later use
print(f"\nLDA model fitted successfully!")
print(f"Model can be used for:")
print(f"  - Visualization (scatter plot)")
print(f"  - Coefficient analysis")
print(f"  - Classification")

# Restore stdout and close file
sys.stdout = original_stdout
output_file.close()
print(f"\nOutput saved to: {output_file_path}")

# Export model and data for use in other scripts
import pickle
model_path = os.path.join(base_dir, 'lda_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(lda, f)
print(f"LDA model saved to: {model_path}")
