import pandas as pd

# List of datasets
datasets = ['drawn_data_cleaned', 'simulated_data_cleaned', 'well_data_cleaned']

for dataset in datasets:
    print(f"\nChecking {dataset}:")
    df = pd.read_csv(f'Cleaned_data/{dataset}.csv')
    print("\nUnique values in 'class' column:")
    print(df['class'].value_counts())
