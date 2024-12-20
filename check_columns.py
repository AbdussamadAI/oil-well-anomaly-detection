import pandas as pd

# Check columns in one of the processed files
df = pd.read_csv('Cleaned_data/balanced_drawn_data_cleaned_processed.csv')
print("Columns in the dataset:")
print(df.columns.tolist())
