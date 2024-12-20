import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import time
import joblib

def load_and_preprocess_data(file_path):
    """Load and preprocess a single dataset"""
    print(f"Loading {file_path}...")
    
    # Read CSV in chunks to handle large files
    chunk_size = 100000
    chunks = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Convert class to numeric and create binary class
        chunk['class'] = pd.to_numeric(chunk['class'], errors='coerce')
        chunk['binary_class'] = (chunk['class'] != 0).astype(int)
        
        # Add source identifier column
        source_name = file_path.split('/')[-1].replace('_cleaned.csv', '')
        chunk['data_source'] = source_name
        
        # Convert timestamp to numeric features
        if 'timestamp' in chunk.columns:
            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
            chunk['hour'] = chunk['timestamp'].dt.hour
            chunk['minute'] = chunk['timestamp'].dt.minute
            chunk['second'] = chunk['timestamp'].dt.second
            chunk = chunk.drop('timestamp', axis=1)
        
        # Drop unnecessary columns
        columns_to_drop = ['source_file', 'class', 'T-JUS-CKGL', 'QGL']
        chunk = chunk.drop([col for col in columns_to_drop if col in chunk.columns], axis=1)
        
        # Handle any remaining non-numeric columns
        numeric_cols = chunk.select_dtypes(include=[np.number]).columns
        chunk = chunk[numeric_cols]
        
        chunks.append(chunk)
        print(f"Processed {len(chunks) * chunk_size} rows...")
    
    print(f"Concatenating chunks for {file_path}...")
    return pd.concat(chunks, ignore_index=True)

def combine_datasets():
    """Combine all three datasets into training and testing sets"""
    datasets = [
        'Cleaned_data/drawn_data_cleaned.csv',
        'Cleaned_data/simulated_data_cleaned.csv',
        'Cleaned_data/well_data_cleaned.csv'
    ]
    
    print("Processing datasets...")
    all_data = []
    for dataset in datasets:
        df = load_and_preprocess_data(dataset)
        print(f"Shape of {dataset}: {df.shape}")
        all_data.append(df)
    
    print("Combining all datasets...")
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined shape: {combined_df.shape}")
    
    # Split features and target
    print("Splitting features and target...")
    X = combined_df.drop(['binary_class'], axis=1)
    y = combined_df['binary_class']
    
    # Calculate class weights
    print("Calculating class weights...")
    class_counts = y.value_counts()
    print("Class distribution:")
    print(class_counts)
    
    total_samples = len(y)
    class_weights = {
        0: total_samples / (2 * class_counts[0]),
        1: total_samples / (2 * class_counts[1])
    }
    
    # Split into train and test sets while maintaining proportions of both classes
    print("Splitting into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the combined datasets
    print("Saving processed datasets...")
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df['binary_class'] = y_train
    train_df.to_csv('combined_train.csv', index=False)
    
    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df['binary_class'] = y_test
    test_df.to_csv('combined_test.csv', index=False)
    
    print("Dataset processing complete!")
    return X_train_scaled, X_test_scaled, y_train, y_test, class_weights

def train_and_evaluate_model(X_train, X_test, y_train, y_test, class_weights, model_name):
    """Train and evaluate a single model with class weights"""
    print(f"\nTraining {model_name}...")
    print(f"Training data shape: {X_train.shape}")
    start_time = time.time()
    
    if model_name == "Random Forest":
        print("Initializing Random Forest...")
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    elif model_name == "XGBoost":
        print("Initializing XGBoost...")
        model = xgb.XGBClassifier(
            scale_pos_weight=class_weights[1]/class_weights[0],
            random_state=42,
            n_jobs=-1,
            verbosity=2
        )
    elif model_name == "LightGBM":
        print("Initializing LightGBM...")
        model = lgb.LGBMClassifier(
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    
    # Train the model
    print("Starting model training...")
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred),
        'Training Time': time.time() - start_time
    }
    
    # Save the model
    print(f"Saving {model_name} model...")
    joblib.dump(model, f'models/combined_{model_name.lower().replace(" ", "_")}.joblib')
    
    return metrics

def main():
    print("Loading and combining datasets...")
    X_train, X_test, y_train, y_test, class_weights = combine_datasets()
    
    print("\nClass distribution in training set:")
    print(pd.Series(y_train).value_counts(normalize=True))
    
    print("\nClass weights:")
    print(class_weights)
    
    # Train all models
    models = ["Random Forest", "XGBoost", "LightGBM"]
    results = []
    
    for model_name in models:
        print(f"\nTraining {model_name}...")
        metrics = train_and_evaluate_model(
            X_train, X_test, y_train, y_test, class_weights, model_name
        )
        results.append(metrics)
    
    # Display results
    results_df = pd.DataFrame(results)
    print("\nFinal Results:")
    print(results_df)
    
    # Save results
    results_df.to_csv('combined_model_results.csv', index=False)
    
    # Find and display the best model
    best_model = results_df.loc[results_df['ROC-AUC'].idxmax()]
    print(f"\nBest Model: {best_model['Model']}")
    print(f"ROC-AUC: {best_model['ROC-AUC']:.6f}")

if __name__ == "__main__":
    main()
