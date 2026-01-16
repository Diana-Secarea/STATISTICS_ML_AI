import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
output_file_path = os.path.join(base_dir, 'lda_assumptions_check_output.txt')
output_file = open(output_file_path, 'w', encoding='utf-8')

# Redirect stdout to both console and file
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, output_file)

# Write header with timestamp
print(f"LDA Assumptions Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
print()

# Load the cleaned dataset (before standardization for normality checks)
print("="*60)
print("STEP 1: Loading cleaned dataset")
print("="*60)
clean_path = os.path.join(base_dir, 'merged_dataset_clean.csv')
merged_clean = pd.read_csv(clean_path)
print(f"Dataset shape: {merged_clean.shape}")

# Load standardized dataset for VIF check
standardized_path = os.path.join(base_dir, 'merged_dataset_standardized.csv')
merged_standardized = pd.read_csv(standardized_path)
print(f"Standardized dataset shape: {merged_standardized.shape}")

# Define indicator columns
indicator_cols = ['GDP_per_capita', 'Unemployment_rate', 'Inflation_CPI', 
                  'Life_expectancy', 'School_enrollment']

# ASSUMPTION 1: Check for Multivariate Normality (Approximate)
print("\n" + "="*60)
print("ASSUMPTION 1: Multivariate Normality Check")
print("="*60)

# 1a. KDE plots by Income Group
print("\n" + "-"*60)
print("1a. Distribution plots (KDE) by Income Group")
print("-"*60)
print("Generating KDE plots for each feature...")

# Create directory for plots if it doesn't exist
plots_dir = os.path.join(base_dir, 'assumptions_plots')
os.makedirs(plots_dir, exist_ok=True)

for col in indicator_cols:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=merged_clean, x=col, hue="Income group", common_norm=False, fill=True)
    plt.title(f'Distribution of {col} by Income Group', fontsize=12, fontweight='bold')
    plt.xlabel(col.replace('_', ' ').title(), fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.legend(title='Income Group', fontsize=9)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, f'kde_{col}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: {plot_path}")

print(f"\nAll KDE plots saved to: {plots_dir}")

# 1b. Shapiro-Wilk test (univariate normality)
print("\n" + "-"*60)
print("1b. Shapiro-Wilk Test for Normality (Univariate)")
print("-"*60)

shapiro_results = []
for col in indicator_cols:
    stat, p = shapiro(merged_clean[col])
    shapiro_results.append({
        'Feature': col,
        'Statistic': stat,
        'p-value': p,
        'Normal (p>0.05)': 'Yes' if p > 0.05 else 'No'
    })
    print(f"{col:25s}: statistic = {stat:.4f}, p-value = {p:.4f} {'[OK] Normal' if p > 0.05 else '[FAIL] Not normal'}")

shapiro_df = pd.DataFrame(shapiro_results)
print("\nSummary:")
print(shapiro_df.to_string(index=False))

# ASSUMPTION 2: Check Homogeneity of Covariance Matrices
print("\n" + "="*60)
print("ASSUMPTION 2: Homogeneity of Covariance Matrices")
print("="*60)
print("Comparing covariance matrices across income groups...")
print()

income_groups = sorted(merged_clean['Income group'].unique())
covariance_matrices = {}

for group in income_groups:
    group_data = merged_clean[merged_clean['Income group'] == group][indicator_cols]
    cov = np.cov(group_data.T)
    covariance_matrices[group] = cov
    print(f"\nCovariance matrix for {group}:")
    print(f"  Shape: {cov.shape}")
    print(f"  Sample size: {len(group_data)}")
    print(cov.round(3))

# Calculate differences between covariance matrices
print("\n" + "-"*60)
print("Covariance Matrix Differences (Frobenius norm)")
print("-"*60)

if len(income_groups) > 1:
    differences = []
    for i, group1 in enumerate(income_groups):
        for j, group2 in enumerate(income_groups):
            if i < j:
                diff = np.linalg.norm(covariance_matrices[group1] - covariance_matrices[group2], 'fro')
                differences.append({
                    'Group 1': group1,
                    'Group 2': group2,
                    'Difference (Frobenius)': diff
                })
                print(f"{group1:25s} vs {group2:25s}: {diff:.3f}")
    
    if differences:
        diff_df = pd.DataFrame(differences)
        avg_diff = diff_df['Difference (Frobenius)'].mean()
        print(f"\nAverage difference: {avg_diff:.3f}")

# ASSUMPTION 3: Check for Multicollinearity
print("\n" + "="*60)
print("ASSUMPTION 3: Multicollinearity Check")
print("="*60)

# 3a. Correlation matrix (already done, but show summary)
print("\n" + "-"*60)
print("3a. Correlation Matrix Summary")
print("-"*60)
correlation_matrix = merged_clean[indicator_cols].corr()
print("Correlation matrix:")
print(correlation_matrix.round(3).to_string())

# Find high correlations (|r| > 0.7)
print("\nHigh correlations (|r| > 0.7):")
high_corr_pairs = []
for i in range(len(indicator_cols)):
    for j in range(i+1, len(indicator_cols)):
        corr_val = correlation_matrix.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append({
                'Feature 1': indicator_cols[i],
                'Feature 2': indicator_cols[j],
                'Correlation': corr_val
            })
            print(f"  {indicator_cols[i]:25s} <-> {indicator_cols[j]:25s}: {corr_val:.3f}")

if not high_corr_pairs:
    print("  None found (all correlations <= 0.7)")

# 3b. Variance Inflation Factor (VIF)
print("\n" + "-"*60)
print("3b. Variance Inflation Factor (VIF)")
print("-"*60)

# Use standardized data for VIF
X_scaled = merged_standardized[indicator_cols].values
X_df = pd.DataFrame(X_scaled, columns=indicator_cols)

vif_data = pd.DataFrame()
vif_data["Feature"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]
vif_data["Multicollinearity"] = vif_data["VIF"].apply(lambda x: "High (VIF>10)" if x > 10 else "Low (VIF<=10)")

print(vif_data.to_string(index=False))

high_vif = vif_data[vif_data['VIF'] > 10]
if len(high_vif) > 0:
    print(f"\n[WARNING] {len(high_vif)} feature(s) with VIF > 10:")
    print(high_vif[['Feature', 'VIF']].to_string(index=False))
else:
    print("\n[OK] No features with VIF > 10 (multicollinearity is acceptable)")

# ASSUMPTION 4: Check Scaling
print("\n" + "="*60)
print("ASSUMPTION 4: Feature Scaling Check")
print("="*60)
print("Verifying that features are standardized (mean ~ 0, std ~ 1)...")
print()

scaling_check = merged_standardized[indicator_cols].describe().T[['mean', 'std']]
scaling_check.columns = ['Mean', 'Std Dev']
scaling_check['Mean_OK'] = scaling_check['Mean'].apply(lambda x: '[OK]' if abs(x) < 0.01 else '[FAIL]')
scaling_check['Std_OK'] = scaling_check['Std Dev'].apply(lambda x: '[OK]' if 0.99 < x < 1.01 else '[FAIL]')

print(scaling_check.round(6).to_string())

all_ok = (scaling_check['Mean_OK'] == '[OK]').all() and (scaling_check['Std_OK'] == '[OK]').all()
if all_ok:
    print("\n[OK] All features are properly standardized!")
else:
    print("\n[WARNING] Some features may not be properly standardized")

# SUMMARY
print("\n" + "="*60)
print("SUMMARY OF LDA ASSUMPTIONS")
print("="*60)

print("\n1. Multivariate Normality:")
normal_count = sum([1 for r in shapiro_results if r['p-value'] > 0.05])
print(f"   - {normal_count}/{len(shapiro_results)} features pass Shapiro-Wilk test (p > 0.05)")
print(f"   - KDE plots saved to: {plots_dir}")

print("\n2. Homogeneity of Covariance Matrices:")
if len(income_groups) > 1 and differences:
    print(f"   - Average covariance difference: {avg_diff:.3f}")
    print("   - Visual inspection of covariance matrices recommended")

print("\n3. Multicollinearity:")
high_vif_count = len(high_vif)
print(f"   - {high_vif_count} feature(s) with VIF > 10")
if high_vif_count == 0:
    print("   [OK] Multicollinearity is acceptable")
else:
    print("   [WARNING] Consider removing or combining highly correlated features")

print("\n4. Feature Scaling:")
if all_ok:
    print("   [OK] All features are properly standardized")
else:
    print("   [WARNING] Some features may need re-standardization")


# Restore stdout and close file
sys.stdout = original_stdout
output_file.close()
print(f"\nOutput saved to: {output_file_path}")
