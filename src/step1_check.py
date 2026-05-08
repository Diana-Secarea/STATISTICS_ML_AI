import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

flights = pd.read_csv(os.path.join(BASE_DIR, 'data', 'flight_data_2024.csv'), low_memory=False)
airports = pd.read_csv(os.path.join(BASE_DIR, 'data', 'Airports-Only.csv'), encoding='latin-1')

# Show the columns
print("Flight Columns:", flights.columns.tolist())
print("Airport Columns:", airports.columns.tolist())

# Test a join
# This links the flight's starting point to the airport's physical location
merged_data = pd.merge(flights, airports, left_on='origin', right_on='IATA')

print("\nSuccess! First 5 rows of merged data:")
print(merged_data[['origin', 'Name', 'Latitude', 'Longitude']].head())