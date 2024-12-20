import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import warnings
import joblib
warnings.filterwarnings('ignore')

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a single model
    """
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        'Training Time': time.time() - start_time
    }
    
    return metrics, model

def process_dataset(file_path, test_size=0.2):
    """
    Process a single dataset
    """
    print(f"\nProcessing {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    X = df.drop('binary_class', axis=1)
    y = df['binary_class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def main():
    # Define models with optimized parameters
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            force_row_wise=True
        )
    }
    
    # Define datasets
    datasets = [
        'Cleaned_data/balanced_drawn_data_cleaned_processed.csv',
        'Cleaned_data/balanced_simulated_data_cleaned_processed.csv',
        'Cleaned_data/balanced_well_data_cleaned_processed.csv'
    ]
    
    # Store results
    all_results = []
    
    # Process each dataset
    for dataset_path in datasets:
        dataset_name = dataset_path.split('/')[-1].replace('_processed.csv', '')
        print(f"\nProcessing {dataset_name}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = process_dataset(dataset_path)
        
        # Train and evaluate each model
        for model_name, model in models.items():
            try:
                metrics, trained_model = train_and_evaluate_model(
                    model, X_train, X_test, y_train, y_test, model_name
                )
                metrics['Dataset'] = dataset_name
                all_results.append(metrics)
                
                # Save the trained model
                model_filename = f"models/{dataset_name}_{model_name.lower().replace(' ', '_')}.joblib"
                joblib.dump(trained_model, model_filename)
                print(f"Saved model to {model_filename}")
                
            except Exception as e:
                print(f"Error training {model_name} on {dataset_name}: {str(e)}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv('simple_model_evaluation_results.csv', index=False)
    
    # Display results
    print("\nFinal Results:")
    pd.set_option('display.max_columns', None)
    print(results_df.round(4))
    
    # Find best model for each dataset
    best_models = results_df.loc[results_df.groupby('Dataset')['ROC-AUC'].idxmax()]
    print("\nBest Models for Each Dataset:")
    print(best_models[['Dataset', 'Model', 'ROC-AUC']].round(4))

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    main()
