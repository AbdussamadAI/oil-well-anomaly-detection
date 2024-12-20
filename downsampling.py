import pandas as pd
import numpy as np
from sklearn.utils import resample
import os

def create_binary_class(df):
    """
    Create a binary classification where class 0 remains 0 and all other classes become 1
    """
    df['binary_class'] = df['class'].apply(lambda x: 1 if x != 0 else 0)
    return df

def downsample_majority_class(df, target_column='binary_class', random_state=42):
    """
    Downsample the larger class to match the smaller class size
    """
    # Get counts of each class
    class_counts = df[target_column].value_counts()
    
    # Identify majority and minority classes
    majority_class = class_counts.index[0]
    minority_class = class_counts.index[1]
    
    # Separate majority and minority classes
    df_majority = df[df[target_column] == majority_class]
    df_minority = df[df[target_column] == minority_class]
    
    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                     replace=False,
                                     n_samples=len(df_minority),
                                     random_state=random_state)
    
    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    
    return df_downsampled

def process_file(input_file):
    """
    Process a single file and save its balanced version
    """
    print(f"\nProcessing {input_file}...")
    
    # Create output filename
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"Cleaned_data/balanced_{base_name}.csv"
    
    # Read and process the file
    print("Reading data...")
    df = pd.read_csv(input_file)
    
    # Create binary classification
    print("Creating binary classification...")
    df = create_binary_class(df)
    
    # Print original distribution
    print("\nOriginal class distribution:")
    print(df['binary_class'].value_counts())
    
    # Perform downsampling
    print("\nPerforming downsampling...")
    df_balanced = downsample_majority_class(df)
    
    # Print balanced distribution
    print("\nBalanced class distribution:")
    print(df_balanced['binary_class'].value_counts())
    
    # Save the balanced dataset
    print(f"\nSaving balanced dataset to '{output_file}'...")
    df_balanced.to_csv(output_file, index=False)
    print(f"Finished processing {input_file}")
    
    # Clear memory
    del df, df_balanced

if __name__ == "__main__":
    # List of files to process
    data_files = [
        "Cleaned_data/drawn_data_cleaned.csv",
        "Cleaned_data/simulated_data_cleaned.csv",
        "Cleaned_data/well_data_cleaned.csv"
    ]
    
    # Process each file separately
    for file in data_files:
        process_file(file)
