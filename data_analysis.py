#!/usr/bin/env python3
"""
Data Loading and Analysis Script for Oil Well Anomaly Detection
This script performs comprehensive data analysis and visualization of the oil well datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_theme(style="whitegrid")

class DataAnalyzer:
    def __init__(self, data_dir="Cleaned_data", sample_size=10000):
        """Initialize DataAnalyzer with data directory path."""
        self.data_dir = Path(data_dir)
        self.sample_size = sample_size
        self.datasets = {
            'drawn': 'balanced_drawn_data_cleaned_processed.csv',
            'simulated': 'balanced_simulated_data_cleaned_processed.csv',
            'well': 'balanced_well_data_cleaned_processed.csv'
        }
        self.feature_cols = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
        self.target_col = 'binary_class'
        
    def load_data(self):
        """Load all datasets and store them in a dictionary."""
        self.data = {}
        for name, file in self.datasets.items():
            df = pd.read_csv(self.data_dir / file)
            # Sample data for visualization
            if len(df) > self.sample_size:
                df = df.sample(n=self.sample_size, random_state=42)
            print(f"\nLoaded {name} dataset: {df.shape}")
            self.data[name] = df
            
    def basic_statistics(self, dataset_name):
        """Calculate and display basic statistics for a dataset."""
        df = self.data[dataset_name]
        print(f"\nBasic Statistics for {dataset_name} dataset:")
        print("\nFeature Statistics:")
        print(df[self.feature_cols].describe())
        
        print("\nClass Distribution:")
        print(df[self.target_col].value_counts(normalize=True))
        
    def plot_feature_distributions(self, dataset_name):
        """Plot distribution of features for normal vs anomaly cases."""
        df = self.data[dataset_name]
        fig, axes = plt.subplots(3, 2, figsize=(15, 20))
        axes = axes.ravel()
        
        for idx, feature in enumerate(self.feature_cols):
            if idx < len(axes):
                sns.kdeplot(data=df, x=feature, hue=self.target_col, ax=axes[idx])
                axes[idx].set_title(f'{feature} Distribution by Class')
        
        plt.tight_layout()
        plt.savefig(f'plots/{dataset_name}_feature_distributions.png')
        plt.close()
        
    def plot_correlation_matrix(self, dataset_name):
        """Plot correlation matrix for features."""
        df = self.data[dataset_name]
        corr = df[self.feature_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Correlation Matrix - {dataset_name}')
        plt.savefig(f'plots/{dataset_name}_correlation_matrix.png')
        plt.close()
        
    def plot_time_series(self, dataset_name, sample_size=1000):
        """Plot time series for each feature."""
        df = self.data[dataset_name].head(sample_size)
        
        fig, axes = plt.subplots(len(self.feature_cols), 1, figsize=(15, 4*len(self.feature_cols)))
        for idx, feature in enumerate(self.feature_cols):
            axes[idx].plot(df.index, df[feature], label='Normal', alpha=0.7)
            axes[idx].scatter(df[df[self.target_col] == 1].index, 
                            df[df[self.target_col] == 1][feature],
                            color='red', label='Anomaly', alpha=0.5)
            axes[idx].set_title(f'{feature} Time Series')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(f'plots/{dataset_name}_time_series.png')
        plt.close()
        
    def analyze_missing_values(self, dataset_name):
        """Analyze missing values in the dataset."""
        df = self.data[dataset_name]
        missing = df[self.feature_cols].isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        print(f"\nMissing Values Analysis for {dataset_name}:")
        for col, count in missing.items():
            print(f"{col}: {count} missing values ({missing_pct[col]:.2f}%)")
            
    def feature_summary(self, dataset_name):
        """Generate summary statistics for features by class."""
        df = self.data[dataset_name]
        normal = df[df[self.target_col] == 0][self.feature_cols]
        anomaly = df[df[self.target_col] == 1][self.feature_cols]
        
        print(f"\nFeature Summary for {dataset_name}:")
        print("\nNormal Cases:")
        print(normal.describe())
        print("\nAnomaly Cases:")
        print(anomaly.describe())
        
    def run_full_analysis(self):
        """Run complete analysis for all datasets."""
        # Create plots directory if it doesn't exist
        Path('plots').mkdir(exist_ok=True)
        
        # Load all datasets
        self.load_data()
        
        # Analyze each dataset
        for dataset_name in self.datasets.keys():
            print(f"\n{'='*50}")
            print(f"Analyzing {dataset_name} dataset")
            print(f"{'='*50}")
            
            # Run all analyses
            self.basic_statistics(dataset_name)
            self.analyze_missing_values(dataset_name)
            self.feature_summary(dataset_name)
            self.plot_feature_distributions(dataset_name)
            self.plot_correlation_matrix(dataset_name)
            self.plot_time_series(dataset_name)

def main():
    """Main function to run the analysis."""
    analyzer = DataAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
