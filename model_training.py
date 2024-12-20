import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, X, y, model_name, dataset_name):
    """
    Evaluate model using stratified k-fold cross validation
    """
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Lists to store metrics
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    fold_times = []
    
    print(f"\nEvaluating {model_name} on {dataset_name}")
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        start_time = time.time()
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        accuracies.append(accuracy_score(y_val, y_pred))
        precisions.append(precision_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))
        roc_aucs.append(roc_auc_score(y_val, y_pred_proba))
        
        fold_times.append(time.time() - start_time)
        
        print(f"Fold {fold} - Accuracy: {accuracies[-1]:.4f}, ROC-AUC: {roc_aucs[-1]:.4f}, Time: {fold_times[-1]:.2f}s")
    
    # Calculate mean metrics
    mean_metrics = {
        'Model': model_name,
        'Dataset': dataset_name,
        'Accuracy': np.mean(accuracies),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1-Score': np.mean(f1_scores),
        'ROC-AUC': np.mean(roc_aucs),
        'Avg Time': np.mean(fold_times)
    }
    
    return mean_metrics

def train_and_evaluate():
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
            tree_method='hist'  # Faster training method
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            force_row_wise=True  # More efficient for smaller datasets
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
        
        # Load data
        df = pd.read_csv(dataset_path)
        X = df.drop('binary_class', axis=1).values
        y = df['binary_class'].values
        
        # Train and evaluate each model
        for model_name, model in models.items():
            try:
                results = evaluate_model(model, X, y, model_name, dataset_name)
                all_results.append(results)
            except Exception as e:
                print(f"Error training {model_name} on {dataset_name}: {str(e)}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_df.to_csv('model_evaluation_results.csv', index=False)
    
    # Display results
    print("\nFinal Results:")
    pd.set_option('display.max_columns', None)
    print(results_df.round(4))
    
    # Find best model for each dataset
    best_models = results_df.loc[results_df.groupby('Dataset')['ROC-AUC'].idxmax()]
    print("\nBest Models for Each Dataset:")
    print(best_models[['Dataset', 'Model', 'ROC-AUC']].round(4))

if __name__ == "__main__":
    train_and_evaluate()
