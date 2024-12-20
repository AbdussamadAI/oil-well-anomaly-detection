import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def create_binary_class(df):
    # Create binary class if it doesn't exist
    if 'binary_class' not in df.columns and 'class' in df.columns:
        df['binary_class'] = (df['class'] == 'GOOD').astype(int)
    return df

def process_features(df):
    # Drop unnecessary columns
    columns_to_drop = ['source_file', 'class', 'T-JUS-CKGL', 'QGL']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Extract time features from timestamp
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day'] = pd.to_datetime(df['timestamp']).dt.day
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df = df.drop('timestamp', axis=1)
    
    # Normalize features
    X = df.drop('binary_class', axis=1)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_scaled['binary_class'] = df['binary_class']
    
    return X_scaled

def process_dataset(dataset):
    # Read the data
    file = f"Cleaned_data/{dataset}.csv"
    df = pd.read_csv(file)
    
    # Create binary class if needed
    df = create_binary_class(df)
    
    # Process features
    df_processed = process_features(df)
    
    # Create output filename
    output_file = file.replace('.csv', '_processed.csv')
    
    # Save processed data
    df_processed.to_csv(output_file, index=False)
    print(f"Saved processed data to {output_file}")
    
    # Display sample of processed data
    print("\nFirst few rows of processed data:")
    print(df_processed.head())
    print("\nFeature statistics:")
    print(df_processed.describe().round(3))

def main():
    # Process both balanced and unbalanced datasets
    datasets = [
        'drawn_data_cleaned',
        'simulated_data_cleaned',
        'well_data_cleaned',
        'balanced_drawn_data_cleaned',
        'balanced_simulated_data_cleaned',
        'balanced_well_data_cleaned'
    ]
    
    for dataset in datasets:
        print(f"\nProcessing {dataset}")
        process_dataset(dataset)

if __name__ == "__main__":
    main()
