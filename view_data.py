import pandas as pd

# Read the balanced well data
df = pd.read_csv('Cleaned_data/balanced_well_data_cleaned.csv')

# Display the first 5 rows with all columns
print("\nFirst 5 rows of balanced well data:")
print(df.head().to_string())
