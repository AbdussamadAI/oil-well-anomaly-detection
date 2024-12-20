import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
import joblib

class ModelTrainer:
    def __init__(self):
        self.base_path = Path('processed_data/split_data')
        self.models_path = Path('models/unified')
        self.models_path.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """
        Load data from all groups:
        - WELL: Training data
        - SIMULATED: Validation data
        - DRAWN: Test data
        """
        print("Loading and preparing datasets...")
        
        # Load WELL data (Training)
        train_data = pd.concat([
            pd.read_csv(self.base_path / 'WELL' / f'fold_{i}/train.csv')
            for i in range(1, 6)
        ])
        
        # Load SIMULATED data (Validation)
        val_data = pd.concat([
            pd.read_csv(self.base_path / 'SIMULATED' / f'fold_{i}/train.csv')
            for i in range(1, 6)
        ])
        
        # Load DRAWN data (Testing)
        test_data = pd.concat([
            pd.read_csv(self.base_path / 'DRAWN' / f'fold_{i}/train.csv')
            for i in range(1, 6)
        ])
        
        # Prepare features and targets
        feature_cols = [col for col in train_data.columns 
                       if col not in ['binary_class', 'timestamp', 'source_file']]
        
        X_train = train_data[feature_cols]
        y_train = train_data['binary_class']
        
        X_val = val_data[feature_cols]
        y_val = val_data['binary_class']
        
        X_test = test_data[feature_cols]
        y_test = test_data['binary_class']
        
        print("\nDataset sizes:")
        print(f"Training (WELL): {len(X_train)} samples")
        print(f"Validation (SIMULATED): {len(X_val)} samples")
        print(f"Testing (DRAWN): {len(X_test)} samples")
        
        print("\nClass distribution:")
        print("Training set:")
        print(y_train.value_counts(normalize=True))
        print("\nValidation set:")
        print(y_val.value_counts(normalize=True))
        print("\nTest set:")
        print(y_test.value_counts(normalize=True))
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols
    
    def train_and_evaluate(self, model_name='xgboost'):
        """
        Train and evaluate a single model using the three datasets
        """
        # Load data
        (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = self.load_data()
        
        # Model configurations
        models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=1,
                tree_method='hist',
                eval_metric='auc'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                random_state=42,
                force_row_wise=True,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=7,
                random_state=42,
                n_jobs=-1
            )
        }
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not supported. Choose from: {list(models.keys())}")
        
        model = models[model_name]
        print(f"\nTraining {model_name}...")
        
        # Train the model
        if model_name == 'xgboost':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        elif model_name == 'lightgbm':
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val)
            )
        else:
            model.fit(X_train, y_train)
        
        # Evaluate on all sets
        datasets = {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        results = {}
        
        for dataset_name, (X, y) in datasets.items():
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]
            
            results[dataset_name] = {
                'auc_roc': roc_auc_score(y, y_pred_proba),
                'confusion_matrix': confusion_matrix(y, y_pred),
                'classification_report': classification_report(y, y_pred, output_dict=True)
            }
            
            print(f"\n{dataset_name.capitalize()} Results:")
            print(f"AUC-ROC: {results[dataset_name]['auc_roc']:.4f}")
            print("\nConfusion Matrix:")
            print(results[dataset_name]['confusion_matrix'])
            print("\nClassification Report:")
            print(classification_report(y, y_pred))
        
        # Save feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.to_csv(
                self.models_path / f'{model_name}_feature_importance.csv',
                index=False
            )
        
        # Save model
        joblib.dump(
            model,
            self.models_path / f'{model_name}_model.joblib'
        )
        
        # Save results
        with open(self.models_path / f'{model_name}_results.txt', 'w') as f:
            for dataset_name, metrics in results.items():
                f.write(f"\n{dataset_name.capitalize()} Results:\n")
                f.write(f"AUC-ROC: {metrics['auc_roc']:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(str(classification_report(y, y_pred)))
        
        return results

def train_models():
    trainer = ModelTrainer()
    models = ['xgboost', 'lightgbm', 'random_forest']
    results = {}
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print('='*50)
        
        results[model_name] = trainer.train_and_evaluate(model_name)
    
    return results

if __name__ == "__main__":
    results = train_models()