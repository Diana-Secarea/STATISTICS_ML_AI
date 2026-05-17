import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
output_file_path = os.path.join(base_dir, 'lda_scatter_plot_output.txt')
output_file = open(output_file_path, 'w', encoding='utf-8')

# Redirect stdout to both console and file
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, output_file)

# Write header with timestamp
print(f"LDA Scatter Plot - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
print()

# Load LDA transformed data (or load processed data and transform)
print("="*60)
print("STEP 1: Loading LDA transformed data")
print("="*60)

# Try to load pre-transformed data, otherwise load processed and transform
lda_data_path = os.path.join(base_dir, 'lda_transformed_data.csv')
if os.path.exists(lda_data_path):
    df = pd.read_csv(lda_data_path)
    print(f"Loaded pre-transformed LDA data from: {lda_data_path}")
else:
    # Load processed data and transform
    processed_path = os.path.join(base_dir, 'merged_dataset_processed.csv')
    df_processed = pd.read_csv(processed_path)
    
    indicator_cols = ['GDP_per_capita', 'Unemployment_rate', 'Inflation_CPI', 
                      'Life_expectancy', 'School_enrollment']
    X = df_processed[indicator_cols].values
    y = df_processed['Income_group_encoded'].values
    
    # Import and use LDA model
    import pickle
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    model_path = os.path.join(base_dir, 'lda_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            lda = pickle.load(f)
        X_lda = lda.transform(X)
    else:
        # Fit new model if not saved
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(X, y)
        X_lda = lda.transform(X)
    
    df = df_processed[['Country Name', 'Country Code', 'Income group', 'Income_group_encoded']].copy()
    df['LD1'] = X_lda[:, 0]
    df['LD2'] = X_lda[:, 1]
    print(f"Transformed data from processed dataset")

print(f"Dataset shape: {df.shape}")
print(f"Number of income groups: {df['Income group'].nunique()}")

# STEP 2: Create Scatter Plot
print("\n" + "="*60)
print("STEP 2: Creating LDA Scatter Plot")
print("="*60)

# Create the scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df,
    x='LD1',
    y='LD2',
    hue='Income group',
    style='Income group',
    s=100,
    alpha=0.7,
    palette='Set2'
)

plt.title('LDA: Countries in Discriminant Space (LD1 vs LD2)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Linear Discriminant 1 (LD1)', fontsize=12)
plt.ylabel('Linear Discriminant 2 (LD2)', fontsize=12)
plt.legend(title='Income Group', title_fontsize=11, fontsize=10, 
           loc='best', framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(base_dir, 'lda_scatter_plot.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Scatter plot saved to: {plot_path}")

# STEP 3: Summary Statistics by Group
print("\n" + "="*60)
print("STEP 3: Summary Statistics by Income Group")
print("="*60)

summary_stats = df.groupby('Income group')[['LD1', 'LD2']].agg(['mean', 'std', 'count'])
print("\nMean and Standard Deviation of LDA components by Income Group:")
print(summary_stats.round(3))

# Calculate separation between groups
print("\n" + "-"*60)
print("Group Centroids (Mean LD1, LD2)")
print("-"*60)
centroids = df.groupby('Income group')[['LD1', 'LD2']].mean()
print(centroids.round(3))

# Calculate distances between centroids
print("\n" + "-"*60)
print("Euclidean Distance Between Group Centroids")
print("-"*60)
from scipy.spatial.distance import pdist, squareform
distances = squareform(pdist(centroids.values))
distance_df = pd.DataFrame(distances, index=centroids.index, columns=centroids.index)
print(distance_df.round(3))


# Restore stdout and close file
sys.stdout = original_stdout
output_file.close()
print(f"\nOutput saved to: {output_file_path}")
