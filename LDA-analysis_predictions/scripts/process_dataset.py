import pandas as pd
import os
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Get the parent directory (project root)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
output_file_path = os.path.join(base_dir, 'process_dataset_output.txt')
output_file = open(output_file_path, 'w', encoding='utf-8')

# Redirect stdout to both console and file
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, output_file)

# Write header with timestamp
print(f"Dataset Processing Output - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
print()

# Load the merged dataset
print("="*60)
print("STEP 1: Loading merged dataset")
print("="*60)
merged_path = os.path.join(base_dir, 'merged_dataset.csv')
merged = pd.read_csv(merged_path)
print(f"Original dataset shape: {merged.shape}")
print(f"Columns: {merged.columns.tolist()}")
print(f"\nMissing values per column:")
print(merged.isnull().sum())

# STEP 1: Cleaning - Drop rows with any missing values
print("\n" + "="*60)
print("STEP 2: Cleaning - Removing rows with missing values")
print("="*60)
merged_clean = merged.dropna()
print(f"Cleaned dataset shape: {merged_clean.shape}")
print(f"Rows removed: {len(merged) - len(merged_clean)}")
print(f"\nIncome group distribution:")
print(merged_clean['Income group'].value_counts().sort_index())

# Generate tables for Data Description section
print("\n" + "="*60)
print("DATA DESCRIPTION TABLES")
print("="*60)

# Table 1: Missing Values Table (Pre-Cleaning)
print("\n" + "-"*60)
print("Table 1: Missing Values (Pre-Cleaning)")
print("-"*60)
missing_data = merged.isnull().sum()
missing_pct = (missing_data / len(merged) * 100).round(1)
missing_table = pd.DataFrame({
    'Variable': missing_data.index,
    'Missing Values': missing_data.values,
    '% Missing': missing_pct.values
})
# Filter to only show variables with missing values
missing_table = missing_table[missing_table['Missing Values'] > 0].sort_values('Missing Values', ascending=False)
print(missing_table.to_string(index=False))

# Table 2: Summary Statistics Table (After Cleaning)
print("\n" + "-"*60)
print("Table 2: Summary Statistics (After Cleaning)")
print("-"*60)
indicator_cols = ['GDP_per_capita', 'Unemployment_rate', 'Inflation_CPI', 
                  'Life_expectancy', 'School_enrollment']
# Use original (non-standardized) values from cleaned dataset
summary_stats = merged_clean[indicator_cols].describe().T[['mean', 'std', 'min', 'max']]
summary_stats.columns = ['Mean', 'Std Dev', 'Min', 'Max']
# Rename index to more readable names
summary_stats.index = ['GDP per capita', 'Unemployment rate', 'Inflation (CPI)', 
                       'Life expectancy', 'School enrollment']
summary_stats = summary_stats.round(2)
print(summary_stats.to_string())

# Table 3: Country Count by Income Group (After Cleaning)
print("\n" + "-"*60)
print("Table 3: Country Count by Income Group (After Cleaning)")
print("-"*60)
income_counts = merged_clean['Income group'].value_counts().sort_index()
income_table = pd.DataFrame({
    'Income Group': income_counts.index,
    'Count': income_counts.values
})
print(income_table.to_string(index=False))

# STEP 2: Standardization - Standardize numeric indicator columns
print("\n" + "="*60)
print("STEP 3: Standardization - Standardizing numeric features")
print("="*60)

# Create a copy for standardization
merged_standardized = merged_clean.copy()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(merged_clean[indicator_cols])

# Create DataFrame with standardized values
merged_standardized[indicator_cols] = X_scaled

print(f"Standardized {len(indicator_cols)} features")
print(f"Mean of standardized features (should be ~0):")
print(merged_standardized[indicator_cols].mean().round(6))
print(f"\nStd of standardized features (should be ~1):")
print(merged_standardized[indicator_cols].std().round(6))

# STEP 3: Encoding - Encode the Income group variable
print("\n" + "="*60)
print("STEP 4: Encoding - Encoding Income group variable")
print("="*60)

# Create a copy for encoding
merged_encoded = merged_standardized.copy()

# Use Categorical with custom order for Income group
income_order = ['Low income', 'Lower middle income', 'Upper middle income', 'High income']
merged_encoded['Income group'] = pd.Categorical(merged_encoded['Income group'], 
                                                 categories=income_order, 
                                                 ordered=True)
y_encoded = merged_encoded['Income group'].cat.codes

# Add encoded column
merged_encoded['Income_group_encoded'] = y_encoded

# Show mapping
print("Income group encoding mapping:")
for i, label in enumerate(income_order):
    print(f"  {i}: {label}")

print(f"\nEncoded values distribution:")
print(pd.Series(y_encoded).value_counts().sort_index())

# Save processed datasets
print("\n" + "="*60)
print("STEP 5: Saving processed datasets")
print("="*60)

# Save cleaned dataset
clean_path = os.path.join(base_dir, 'merged_dataset_clean.csv')
merged_clean.to_csv(clean_path, index=False)
print(f"Cleaned dataset saved to: {clean_path}")

# Save standardized dataset
standardized_path = os.path.join(base_dir, 'merged_dataset_standardized.csv')
merged_standardized.to_csv(standardized_path, index=False)
print(f"Standardized dataset saved to: {standardized_path}")

# Save final encoded dataset
encoded_path = os.path.join(base_dir, 'merged_dataset_processed.csv')
merged_encoded.to_csv(encoded_path, index=False)
print(f"Final processed dataset (with encoding) saved to: {encoded_path}")

# Also save the scaler and label encoder info for reference
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Original dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
print(f"After cleaning: {merged_clean.shape[0]} rows, {merged_clean.shape[1]} columns")
print(f"After standardization: {merged_standardized.shape[0]} rows, {merged_standardized.shape[1]} columns")
print(f"Final processed: {merged_encoded.shape[0]} rows, {merged_encoded.shape[1]} columns")
print(f"\nAll processing steps completed successfully!")

# Restore stdout and close file
sys.stdout = original_stdout
output_file.close()
print(f"\nOutput saved to: {output_file_path}")
