import pandas as pd
import os

# Get the parent directory (project root)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datasets_dir = os.path.join(base_dir, 'datasets')

# Load your income group labels
income_df = pd.read_csv(os.path.join(datasets_dir, "Income_group.csv"))[['Country Name', 'Country Code', 'Income group']]

# Load indicator files
gdp = pd.read_csv(os.path.join(datasets_dir, "GDP_per_capita.csv"))[['Country Name', 'Country Code', '2022']]
gdp = gdp.rename(columns={'2022': 'GDP_per_capita'})

unemp = pd.read_csv(os.path.join(datasets_dir, "Unemployment.csv"))[['Country Name', 'Country Code', '2022']]
unemp = unemp.rename(columns={'2022': 'Unemployment_rate'})

inflation = pd.read_csv(os.path.join(datasets_dir, "Inflation_CPI.csv"))[['Country Name', 'Country Code', '2022']]
inflation = inflation.rename(columns={'2022': 'Inflation_CPI'})

life = pd.read_csv(os.path.join(datasets_dir, "Life_expectancy.csv"))[['Country Name', 'Country Code', '2022']]
life = life.rename(columns={'2022': 'Life_expectancy'})

edu = pd.read_csv(os.path.join(datasets_dir, "School_enrollment.csv"))[['Country Name', 'Country Code', '2022']]
edu = edu.rename(columns={'2022': 'School_enrollment'})

# Merge everything using both 'Country Name' and 'Country Code'
dfs = [gdp, unemp, inflation, life, edu]
merged = income_df
for df in dfs:
    merged = pd.merge(merged, df, on=['Country Name', 'Country Code'], how='inner')  # inner keeps only countries with complete data

# Reorder columns: Country Name, Country Code, GDP, Unemployment, Inflation, Life Expectancy, School Enrollment, Income group
column_order = ['Country Name', 'Country Code', 'GDP_per_capita', 'Unemployment_rate', 
                'Inflation_CPI', 'Life_expectancy', 'School_enrollment', 'Income group']
merged = merged[column_order]

# Display the result
print(f"Merged dataset shape: {merged.shape}")
print(f"\nColumns: {merged.columns.tolist()}")
print(f"\nFirst 10 rows:")
print(merged.head(10))
print(f"\nSummary statistics:")
print(merged.describe())

# Save the merged dataset
output_path = os.path.join(base_dir, 'merged_dataset.csv')
merged.to_csv(output_path, index=False)
print(f"\nMerged dataset saved to: {output_path}")
