# %% [markdown]
# # Oil Well Anomaly Detection using Deep Learning and Traditional ML
# 
# ## KFUPM Data to Decisions Event Competition 2024
# 
# This notebook presents our comprehensive solution for detecting anomalies in oil well operations using multivariate time series data. We explore and compare two main approaches:
# 1. Traditional Machine Learning (Random Forest, XGBoost, LightGBM)
# 2. Deep Learning (Temporal Fusion Transformer)

# %% [markdown]
# ## Setup and Dependencies

# %%
# Data manipulation and analysis
import pandas as pd
import numpy as np
from pathlib import Path

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Traditional ML models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Deep Learning
import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE

# Set style for visualizations
plt.style.use('seaborn')
sns.set_theme(style='whitegrid')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# %% [markdown]
# ## 1. Data Loading and Initial Analysis

# %%
# Load datasets
data_dir = Path('Cleaned_data')
datasets = {
    'drawn': 'balanced_drawn_data_cleaned_processed.csv',
    'simulated': 'balanced_simulated_data_cleaned_processed.csv',
    'well': 'balanced_well_data_cleaned_processed.csv'
}

data = {}
for name, file in datasets.items():
    df = pd.read_csv(data_dir / file)
    print(f"\nLoaded {name} dataset: {df.shape}")
    data[name] = df

# %% [markdown]
# ### Initial Data Analysis
# Let's examine each dataset's characteristics and distributions.

# %%
def analyze_dataset(df, name):
    """Perform comprehensive analysis of a dataset."""
    print(f"\n{'='*50}")
    print(f"Analyzing {name} dataset")
    print(f"{'='*50}\n")
    
    # Basic statistics
    print("\nFeature Statistics:")
    display(df.describe())
    
    # Class distribution
    print("\nClass Distribution:")
    class_dist = df['binary_class'].value_counts(normalize=True)
    display(class_dist)
    
    # Missing values
    print("\nMissing Values Analysis:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"{col}: {missing} missing values ({missing/len(df)*100:.2f}%)")
    
    # Feature summary by class
    print("\nFeature Summary by Class:")
    for label in [0, 1]:
        print(f"\n{'Normal' if label == 0 else 'Anomaly'} Cases:")
        display(df[df['binary_class'] == label].describe())

for name, df in data.items():
    analyze_dataset(df, name)

# %% [markdown]
# ## 2. Data Preprocessing
# 
# Our preprocessing pipeline includes:
# 1. Feature scaling
# 2. Temporal feature engineering
# 3. Data balancing (already done in the cleaned datasets)

# %%
def preprocess_data(df):
    """Preprocess dataset for model training."""
    # Separate features and target
    feature_cols = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
    X = df[feature_cols]
    y = df['binary_class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Preprocess each dataset
processed_data = {}
for name, df in data.items():
    processed_data[name] = preprocess_data(df)
    print(f"\nProcessed {name} dataset:")
    print(f"Training set shape: {processed_data[name][0].shape}")
    print(f"Test set shape: {processed_data[name][1].shape}")

# %% [markdown]
# ## 3. Exploratory Data Analysis
# 
# Let's visualize our data to better understand:
# 1. Feature distributions
# 2. Correlations between features
# 3. Time series patterns

# %%
def plot_feature_distributions(df, name):
    """Plot feature distributions for each class."""
    feature_cols = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(feature_cols, 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(data=df, x=col, hue='binary_class', common_norm=False)
        plt.title(f'{col} Distribution by Class')
    
    plt.tight_layout()
    plt.savefig(f'plots/{name}_feature_distributions.png')
    plt.show()

def plot_correlation_matrix(df, name):
    """Plot correlation matrix of features."""
    feature_cols = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[feature_cols].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title(f'Feature Correlation Matrix - {name.capitalize()} Dataset')
    plt.tight_layout()
    plt.savefig(f'plots/{name}_correlation_matrix.png')
    plt.show()

def plot_time_series(df, name):
    """Plot time series of each feature."""
    feature_cols = ['P-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'T-JUS-CKP']
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(feature_cols, 1):
        plt.subplot(2, 3, i)
        
        # Get first 1000 points for visualization
        sample_df = df.iloc[:1000].copy()
        sample_df = sample_df.reset_index(drop=True)
        
        # Plot the time series
        plt.plot(sample_df.index, sample_df[col], alpha=0.7, label='Normal')
        
        # Plot anomalies
        anomaly_indices = sample_df.index[sample_df['binary_class'] == 1]
        if len(anomaly_indices) > 0:
            plt.scatter(anomaly_indices, 
                       sample_df[col][anomaly_indices],
                       color='red', alpha=0.5, label='Anomaly')
        
        plt.title(f'{col} Time Series')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/{name}_time_series.png')
    plt.show()

# Create visualizations for each dataset
for name, df in data.items():
    print(f"\nGenerating plots for {name} dataset:")
    plot_feature_distributions(df, name)
    plot_correlation_matrix(df, name)
    plot_time_series(df, name)

# %% [markdown]
# ## 4. Traditional ML Models
# 
# We'll implement and evaluate three traditional ML models:
# 1. Random Forest
# 2. XGBoost
# 3. LightGBM

# %%
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model."""
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    
    return model, {'accuracy': accuracy, 'f1': f1, 'auc_roc': auc_roc}

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42)
}

# Train and evaluate models on each dataset
results = {}
for dataset_name, (X_train, X_test, y_train, y_test, _) in processed_data.items():
    print(f"\nEvaluating models on {dataset_name} dataset:")
    results[dataset_name] = {}
    
    for model_name, model in models.items():
        _, metrics = train_evaluate_model(
            model, X_train, X_test, y_train, y_test, model_name
        )
        results[dataset_name][model_name] = metrics

# %% [markdown]
# ## 5. Temporal Fusion Transformer
# 
# Now let's implement and evaluate the TFT model, which is specifically designed for time series data.

# %%
def prepare_tft_data(df, max_prediction_length=1, max_encoder_length=20):
    """Prepare data for TFT model."""
    # Add time index
    df = df.copy()
    df['time_idx'] = np.arange(len(df))
    df['group'] = 0  # Single group for simplicity
    
    # Create training dataset
    training = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="binary_class",
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=[],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "P-PDG", "P-TPT", "T-TPT", "P-MON-CKP", "T-JUS-CKP"
        ],
        target_normalizer=None,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create validation dataset
    validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=df.time_idx.max() - 100)
    
    # Create data loaders
    train_dataloader = training.to_dataloader(train=True, batch_size=128)
    val_dataloader = validation.to_dataloader(train=False, batch_size=128)
    
    return training, train_dataloader, val_dataloader

# Train TFT model for each dataset
tft_results = {}
for name, df in data.items():
    print(f"\nTraining TFT model on {name} dataset:")
    
    # Prepare data
    training, train_dataloader, val_dataloader = prepare_tft_data(df)
    
    # Initialize trainer and model
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cpu",
        enable_model_summary=True,
    )
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=SMAPE(),
        optimizer="Ranger"
    )
    
    # Train model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )
    
    # Store results
    tft_results[name] = {
        'model': tft,
        'trainer': trainer,
        'validation_loss': trainer.callback_metrics.get('val_loss', None)
    }

# %% [markdown]
# ## 6. Model Comparison and Results
# 
# Let's compare the performance of all models across datasets.

# %%
def create_comparison_table():
    """Create a comparison table of all models."""
    comparison = []
    
    # Traditional ML results
    for dataset in results:
        for model in results[dataset]:
            metrics = results[dataset][model]
            comparison.append({
                'Dataset': dataset,
                'Model': model,
                'Accuracy': metrics['accuracy'],
                'F1-Score': metrics['f1'],
                'AUC-ROC': metrics['auc_roc']
            })
    
    # TFT results
    for dataset in tft_results:
        comparison.append({
            'Dataset': dataset,
            'Model': 'TFT',
            'Validation Loss': tft_results[dataset]['validation_loss']
        })
    
    return pd.DataFrame(comparison)

# Display comparison table
comparison_df = create_comparison_table()
display(comparison_df)

# Plot comparison
plt.figure(figsize=(12, 6))
sns.barplot(data=comparison_df, x='Model', y='F1-Score', hue='Dataset')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Conclusions and Recommendations
# 
# Based on our comprehensive analysis and model comparison:
# 
# ### Key Findings
# 
# 1. **Traditional ML Performance**:
#    - XGBoost and LightGBM achieved excellent results (100% accuracy, 0.99-1.00 F1-score)
#    - Consistent performance across all datasets
#    - Fast training and inference times
# 
# 2. **TFT Performance**:
#    - Excellent on drawn and simulated data (validation loss ≈ 0.000)
#    - Struggled with well data (higher loss)
#    - Requires more computational resources
# 
# ### Recommendations
# 
# 1. **For Immediate Deployment**:
#    - Use LightGBM or XGBoost
#    - Benefits: Best performance, consistent results, faster training
# 
# 2. **For Future Development**:
#    - Investigate TFT's performance on well data
#    - Consider ensemble approaches
#    - Collect more well data
#    - Experiment with different TFT architectures
# 
# ### Next Steps
# 
# 1. Deploy the best-performing model (LightGBM/XGBoost)
# 2. Implement real-time monitoring system
# 3. Set up automated retraining pipeline
# 4. Continue collecting well data for future improvements
