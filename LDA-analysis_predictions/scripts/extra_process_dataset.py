import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def create_correlation_heatmap(dataset_path=None, save_path=None, figsize=(10, 8), 
                                cmap='coolwarm', annot=True, fmt='.2f'): 
    # Get the parent directory (project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load dataset
    if dataset_path is None:
        dataset_path = os.path.join(base_dir, 'merged_dataset_processed.csv')
    
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Select numeric indicator columns (exclude encoded target variable for correlation)
    indicator_cols = ['GDP_per_capita', 'Unemployment_rate', 'Inflation_CPI', 
                      'Life_expectancy', 'School_enrollment']
    
    # Check if columns exist
    available_cols = [col for col in indicator_cols if col in df.columns]
    if len(available_cols) != len(indicator_cols):
        missing = set(indicator_cols) - set(available_cols)
        print(f"Warning: Missing columns: {missing}")
        print(f"Using available columns: {available_cols}")
    
    # Calculate correlation matrix
    correlation_matrix = df[available_cols].corr()
    
    # Create readable column names for display
    display_names = {
        'GDP_per_capita': 'GDP per capita',
        'Unemployment_rate': 'Unemployment rate',
        'Inflation_CPI': 'Inflation (CPI)',
        'Life_expectancy': 'Life expectancy',
        'School_enrollment': 'School enrollment'
    }
    
    # Rename columns for display
    correlation_matrix_display = correlation_matrix.rename(
        index=display_names, 
        columns=display_names
    )
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        correlation_matrix_display,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
    )
    
    plt.title('Correlation Matrix Heatmap - Economic Indicators', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    # Save the figure
    if save_path is None:
        save_path = os.path.join(base_dir, 'correlation_heatmap.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved to: {save_path}")
    
    # Close the figure to free memory
    plt.close()
    
    # Print correlation matrix as table
    print("\n" + "="*60)
    print("Correlation Matrix")
    print("="*60)
    print(correlation_matrix_display.round(3).to_string())
    
    return plt.gcf()


if __name__ == "__main__":
    # Run the function when script is executed directly
    create_correlation_heatmap()
