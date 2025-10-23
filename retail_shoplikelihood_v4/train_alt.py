import logging
import sys
import os
from google.cloud import storage
from google.cloud import bigquery
from google.api_core.exceptions import ServerError, TooManyRequests
from pyarrow import parquet
import gcsfs
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from joblib import dump
import time
import optuna
from functools import partial
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

hyperparameters_tuning = False 
default_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'class_weight': 'balanced',  # Use balanced class weights
    'n_jobs': -1,  # Use all available cores
    'random_state': 42
}

def save_model(model_path, model_name, model):
    """Save model to gcs bucket path."""
    fs = gcsfs.GCSFileSystem()
    model_gcs_path = model_path + model_name
    with fs.open(model_gcs_path, 'wb') as f:
        dump(model, f, compress=3)

    logger.info(f"Model saved to: {model_gcs_path}")

def batch_upload(batch_size, df, table_name, source_format, bq_client):
    """Upload DataFrame to BigQuery in chunks."""
    print(f"Uploading {len(df)} rows to {table_name} in batches of {batch_size}...")
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        chunk = df.iloc[start:end]
        write_disposition = "WRITE_TRUNCATE" if start == 0 else "WRITE_APPEND"
        job_config = bigquery.LoadJobConfig(source_format=source_format, write_disposition=write_disposition)
        job = bq_client.load_table_from_dataframe(dataframe=chunk,
                                                    destination=table_name,
                                                    job_config=job_config
                                                )
        job.result()  # Waits for the job to complete
        print(f"Uploaded rows {start} to {end} to {table_name}")
    print("All batches uploaded successfully.")

def objective(trial, X, y):
    """Optuna objective function for RandomForest optimization."""
    
    # Define the hyperparameter search space
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'class_weight': 'balanced',  # Keep this fixed since it works well
        'n_jobs': -1,
        'random_state': 42
    }
    
    # Create classifier with trial parameters
    clf = RandomForestClassifier(**params)
    
    # Use cross-validation to evaluate
    scores = cross_val_score(
        clf, 
        X, 
        y, 
        cv=3,  # 3-fold CV to balance speed and reliability
        scoring='roc_auc',  # Use ROC-AUC as optimization metric
        n_jobs=-1
    )
    
    return scores.mean()

def hyperparameter_optimization(X_train, y_train):
    logger.info("Starting hyperparameter optimization with Optuna...")
    # Create study object
    study = optuna.create_study(
        direction="maximize",
        study_name="rf_optimization",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    n_trials = 20  # Adjust based on your time constraints
    study.optimize(
        partial(objective, X=X_train, y=y_train),
        n_trials=n_trials,
        n_jobs=1  # Using 1 here because RF already uses parallel processing
    )
    
    # Log optimization results
    logger.info("Best trial:")
    logger.info(f"  Value: {study.best_trial.value:.4f}")
    logger.info("  Params: ")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Train final model with best parameters
    best_params = study.best_trial.params
    best_params.update({
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42
    })
    logger.info(f"Best parameters: {best_params}")
    return best_params

def evaluate_classifier(model, model_name, X_train, X_val, X_test, y_train, y_val, y_test):
    """Train and evaluate a classifier with comprehensive metrics."""
    logger.info(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_predictions = model.predict(X_val)
    val_prob_predictions = model.predict_proba(X_val)[:, 1]
    
    logger.info(f"\n{model_name} Validation Set Metrics:")
    logger.info(classification_report(y_val, val_predictions))
    logger.info(f"Validation AUC-ROC: {roc_auc_score(y_val, val_prob_predictions):.4f}")
    
    # Evaluate on test set
    test_predictions = model.predict(X_test)
    test_prob_predictions = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"\n{model_name} Test Set Metrics:")
    logger.info(classification_report(y_test, test_predictions))
    logger.info(f"Test AUC-ROC: {roc_auc_score(y_test, test_prob_predictions):.4f}")
    
    return {
        'model': model,
        'val_auc': roc_auc_score(y_val, val_prob_predictions),
        'test_auc': roc_auc_score(y_test, test_prob_predictions),
        'val_predictions': val_predictions,
        'test_predictions': test_predictions,
        'val_probabilities': val_prob_predictions,
        'test_probabilities': test_prob_predictions
    }

def create_deep_learning_model(input_dim, scale_pos_weight):
    """Create a deep neural network for binary classification."""
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    return model

class KerasClassifierWrapper:
    """Wrapper class to make Keras model compatible with sklearn interface."""
    def __init__(self, input_dim, scale_pos_weight):
        self.model = create_deep_learning_model(input_dim, scale_pos_weight)
        self.scaler = StandardScaler()
        self.scale_pos_weight = scale_pos_weight
        
    def fit(self, X, y):
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert y to numpy array if it's not already
        y = np.asarray(y)
        
        # Calculate class weights using numpy unique values
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError(f"Expected binary classification, got {len(unique_classes)} classes")
            
        # Create class weights dictionary using integer keys
        class_weights = {
            int(0): 1.0,
            int(1): float(self.scale_pos_weight)
        }
        
        logger.info(f"Class weights being used: {class_weights}")
        logger.info(f"Unique classes in y: {unique_classes}")
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_auc',
            patience=5,
            mode='max',
            restore_best_weights=True
        )
        
        try:
            # Fit the model with explicit error handling
            self.model.fit(
                X_scaled, 
                y,
                epochs=50,
                batch_size=256,
                validation_split=0.2,
                class_weight=class_weights,
                callbacks=[early_stopping],
                verbose=1
            )
        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            logger.error(f"Input shapes - X: {X_scaled.shape}, y: {y.shape}")
            logger.error(f"y unique values: {np.unique(y, return_counts=True)}")
            raise
            
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return (self.model.predict(X_scaled) > 0.5).astype(int)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled)
        return np.column_stack([1 - preds, preds])

if __name__ == "__main__":

    logger.info("Loading environment variables...")
    customer_purchase_labels_path = os.environ.get('customer_purchase_labels_path')
    gcs_embeddings_output_path = os.environ.get('gcs_embeddings_output_path')
    last_week_emb_mapping_table = os.environ.get('last_week_emb_mapping_table')
    predictions_output_table = os.environ.get('predictions_output_table')

    last_week_identifier = os.environ.get('last_week_identifier')
    model_output_path = os.environ.get('model_output_path')

    logger.info("Starting bigquery client and GCS file system...")
    bq_project = os.environ.get('bq_project')
    bq_client = bigquery.Client(project=bq_project)
    fs               = gcsfs.GCSFileSystem()
    
    logger.info(f"Loading customer purchase labels from: {customer_purchase_labels_path}")
    customer_purchase_labels = [_path for _path in fs.glob(os.path.join(customer_purchase_labels_path, "*.parquet"))]
    customer_labels_ds       = parquet.ParquetDataset(customer_purchase_labels, filesystem=fs)
    customer_labels_pd_df    = customer_labels_ds.read().to_pandas()
    logger.info(f"Loaded {len(customer_labels_pd_df)} customer purchase labels.")

    logger.info(f"Loading LightFM embeddings from: {gcs_embeddings_output_path}")
    embeddings_files = [_path for _path in fs.glob(os.path.join(gcs_embeddings_output_path, "*.parquet"))]
    embedding_ds = parquet.ParquetDataset(embeddings_files, filesystem=fs)
    embedding_df = embedding_ds.read().to_pandas()
    logger.info(f"Total embedding rows loaded: {len(embedding_df)}")


    model_features_df = embedding_df.merge(customer_labels_pd_df[['epsilon_id', 'purchase_flag']], on='epsilon_id', how='left')
    model_features_df["purchase_flag"] = model_features_df["purchase_flag"].fillna('N')

    logger.info("Model features shape: ", model_features_df.shape)
    logger.info(model_features_df.head())
    logger.info(model_features_df.groupby('purchase_flag').count())
    
    # =============================================
    # Prepare data for training
    # =============================================
     # Get feature columns (latent_0 to latent_199)
    feature_cols = [f'latent_{i}' for i in range(200)]
    
    # Verify all feature columns exist
    if not all(col in model_features_df.columns for col in feature_cols):
        missing_cols = [col for col in feature_cols if col not in model_features_df.columns]
        logger.error(f"Missing feature columns: {missing_cols}")
        raise ValueError("Missing expected feature columns")

    # Prepare X (features) and y (target)
    X = model_features_df[feature_cols]
    y = (model_features_df['purchase_flag'] == 'Y').astype(int)  # Convert to binary

    # Create train, validation, and test sets (60/20/20 split)
    logger.info("Splitting data into train, validation, and test sets...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # 20% for test set
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)  # 0.25 of 80% is 20%

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")

    
    
    if hyperparameters_tuning:
        best_params = hyperparameter_optimization(X_train, y_train)
    else:
        logger.info("Using default hyperparameters for RandomForestClassifier.")
        best_params = default_params

    # Calculate class weight for XGBoost and LightGBM
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Initialize all models
    models = {
        'RandomForest': RandomForestClassifier(
            **default_params
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42
        ),
        'DeepLearning': KerasClassifierWrapper(
            input_dim=X_train.shape[1],
            scale_pos_weight=scale_pos_weight
        )
    }

    # Evaluate all models
    results = {}
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        results[model_name] = evaluate_classifier(
            model, model_name, 
            X_train, X_val, X_test, 
            y_train, y_val, y_test
        )

    # Compare models
    logger.info("\nModel Comparison Summary:")
    comparison_df = pd.DataFrame({
        model_name: {
            'Validation AUC-ROC': results[model_name]['val_auc'],
            'Test AUC-ROC': results[model_name]['test_auc']
        } for model_name in models.keys()
    }).round(4)
    
    logger.info("\n" + str(comparison_df))

    # Select best model based on validation AUC-ROC
    best_model_name = max(results.items(), key=lambda x: x[1]['val_auc'])[0]
    best_model = results[best_model_name]['model']
    logger.info(f"\nBest performing model: {best_model_name}")

    # Save best model
    model_name = f'best_model_{best_model_name}_{last_week_identifier}.joblib'
    save_model(model_path=model_output_path, model_name=model_name, model=best_model)

    # Generate predictions using best model
    model_features_df['predicted_purchase_prob'] = best_model.predict_proba(X)[:, 1]
    
    batch_size = 400000  
    # Save predictions with customer IDs
    predictions_df = model_features_df[['epsilon_id', 'lightfm_id', 'user_id', 'predicted_purchase_prob']]
    batch_upload(
        batch_size=batch_size, 
        df=predictions_df, 
        table_name=predictions_output_table, 
        source_format=bigquery.SourceFormat.PARQUET, 
        bq_client=bq_client
    )
    
    logger.info(f"Predictions saved to table: {predictions_output_table}")
    logger.info("Model comparison and evaluation completed successfully.")