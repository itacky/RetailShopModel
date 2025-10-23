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
from lightgbm import LGBMClassifier, early_stopping
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from datetime import datetime
import shap
from scipy.special import expit  # Sigmoid function

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

def save_features(features_path, features_name, features):
    """Save features as a Parquet file to GCS bucket path."""
    fs = gcsfs.GCSFileSystem()
    features_gcs_path = os.path.join(features_path, features_name)

    # Save features as a Parquet file
    with fs.open(features_gcs_path, 'wb') as f:
        features.to_parquet(f, index=False, engine='pyarrow')  # Use pyarrow or fastparquet as the engine

    logger.info(f"Features saved as Parquet to: {features_gcs_path}")


def batch_upload_to_gcs(df, output_path, batch_size, file_name, fs):
    """Upload DataFrame to GCS in parquet format with batching."""
    logger.info(f"Uploading {len(df)} rows to {output_path} in batches of {batch_size}...")
    
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        chunk = df.iloc[start:end]
        
        # Create unique filename for this chunk
        chunk_filename = f"{file_name}_{start}_{end}.parquet"
        full_path = os.path.join(output_path, chunk_filename)
        
        logger.info(f"Writing chunk {start} to {end} to {full_path}")
        with fs.open(full_path, 'wb') as f:
            chunk.to_parquet(f, index=False)
    
    logger.info("All batches uploaded successfully to GCS.")   


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


def load_from_bigquery(query, bq_client):
    return bq_client.query(query).to_dataframe()


def load_reco_reference_df(bq_client):
    print("Loading deal week reference table...")
    query = """ 
    SELECT 
        start_date, 
        end_date, 
        deal_yr, 
        deal_wk,
        CONCAT(CAST(deal_yr AS STRING), '_3', LPAD(CAST(deal_wk AS STRING), 2, '0')) AS week_identifier,
        deal_season_num
    from `dp-lore.loyalty_reco.prd_slpv3_date_ref` 
    """
    return load_from_bigquery(query=query, bq_client=bq_client)


def rename_pivot_columns(pivoted_df, reco_reference_pd, current_week_identifier, suffix:str, removal_cols:list=None):
    logger.info(f"Renaming pivoted columns with suffix '{suffix}'")
    col_reference = {}
    current_start_datetime = reco_reference_pd[reco_reference_pd["week_identifier"] == current_week_identifier].start_date.iloc[0]
    pivoted_cols = pivoted_df.columns
    if removal_cols:
        pivoted_cols = pivoted_df.columns.drop(removal_cols)

    for col in pivoted_cols:
        ref_start_datetime = reco_reference_pd[reco_reference_pd["week_identifier"] == col].start_date.iloc[0]
        delta = current_start_datetime - ref_start_datetime
        weeks_back = delta.days/7
        num_weeks_back = str(int(weeks_back))+suffix

        col_reference[col] = num_weeks_back

    # Rename columns based on the mapping    
    pivoted_df.rename(columns=col_reference, inplace=True)
    pivoted_df = pivoted_df.drop(columns=[f'0{suffix}'], errors='ignore')
    logger.info(f"renamed columns: {[pivoted_df.columns]}")
    return pivoted_df


def objective(trial, X, y):
    """Optuna objective function for RandomForest optimization."""
    
    # Define the hyperparameter search space
    # params = {
    #     'n_estimators': trial.suggest_int('n_estimators', 50, 300),
    #     'max_depth': trial.suggest_int('max_depth', 5, 30),
    #     'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    #     'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
    #     'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    #     'class_weight': 'balanced',  # Keep this fixed since it works well
    #     'n_jobs': -1,
    #     'random_state': 42
    # }
    
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', -1, 30),  # -1 means no limit
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),  # Handle class imbalance
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Split data into training and validation sets for early stopping
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X, y, test_size=0.5, random_state=42, stratify=y
    )

    # Detect binary and numeric columns
    binary_cols = [col for col in X_train_split.columns if set(X_train_split[col].dropna().unique()).issubset({0, 1})]
    numeric_cols = [col for col in X_train_split.columns if col not in binary_cols]

    # Fit scaler only on training data
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('bin', 'passthrough', binary_cols)
    ])
    # Scale the training data
    X_train_scaled = preprocessor.fit_transform(X_train_split)

    # Scale validation and test data using the same parameters
    X_val_scaled = preprocessor.transform(X_val_split)

    # Create classifier with trial parameters
    # clf = RandomForestClassifier(**params)
    clf = LGBMClassifier(**params)
    
    # Fit the model with early stopping
    clf.fit(
        X_train_scaled, 
        y_train_split, 
        eval_set=[(X_val_scaled, y_val_split)], 
        eval_metric='auc', 
        callbacks=[early_stopping(stopping_rounds=5)]  # Stop if validation AUC does not improve for 50 rounds
    )
    
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

def hyperparameter_optimization(X, y):
    logger.info("Starting hyperparameter optimization with Optuna...")
    # Create study object
    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_optimization",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Run optimization
    n_trials = 14  # Adjust based on your time constraints
    study.optimize(
        partial(objective, X=X, y=y),
        n_trials=n_trials,
        n_jobs=-1  # Using 1 here because RF already uses parallel processing
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
        # 'class_weight': 'balanced',
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

def create_deep_learning_model(input_dim, scale_pos_weight, learning_rate=0.001):
    """Create a deep neural network for binary classification."""
    model = Sequential([
        Dense(512, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    return model

class KerasClassifierWrapper:
    """Wrapper class to make Keras model compatible with sklearn interface."""
    def __init__(self, input_dim, scale_pos_weight, learning_rate=0.001):
        self.model = create_deep_learning_model(input_dim, scale_pos_weight, learning_rate)
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

def shap_analysis(model, model_name, X_train_scaled, X_val_scaled, feature_cols):
    """Perform SHAP value analysis for a given model and log results."""
    logger.info(f"\nPerforming SHAP analysis for {model_name}...")

    # Use a randomized subset of the validation data for SHAP analysis
    shap_sample_size = 10000
    if len(X_val_scaled) > shap_sample_size:
        random_indices = np.random.choice(len(X_val_scaled), shap_sample_size, replace=False)
        X_val_sample = X_val_scaled[random_indices]
    else:
        X_val_sample = X_val_scaled

    # Check if the model supports SHAP
    if hasattr(model, "predict_proba") or hasattr(model, "predict"):
        try:
            # Use TreeExplainer for tree-based models
            if model_name in ["RandomForest", "XGBoost", "LightGBM"]:
                explainer = shap.TreeExplainer(model, model_output="raw")  # Use raw output
                shap_values_raw = explainer.shap_values(X_val_sample)

                # Convert raw SHAP values to probabilities
                base_value = explainer.expected_value  # Base value (log-odds)
                shap_values_class_1 = expit(base_value + shap_values_raw)  # Convert to probabilities for class 1
                shap_values_class_0 = 1 - shap_values_class_1  # Probabilities for class 0

                # SHAP summary for class 1 (positive class)
                logger.info(f"SHAP summary for class 1 (positive class):")
                shap_summary_class_1 = np.mean(np.abs(shap_values_class_1), axis=0)
                shap_summary_df_1 = pd.DataFrame({
                    "feature": feature_cols,
                    "mean_abs_shap_value": shap_summary_class_1
                }).sort_values("mean_abs_shap_value", ascending=False)
                logger.info("\n" + str(shap_summary_df_1.head(20)))  # Log top 20 features

                # SHAP summary for class 0 (negative class)
                logger.info(f"SHAP summary for class 0 (negative class):")
                shap_summary_class_0 = np.mean(np.abs(shap_values_class_0), axis=0)
                shap_summary_df_0 = pd.DataFrame({
                    "feature": feature_cols,
                    "mean_abs_shap_value": shap_summary_class_0
                }).sort_values("mean_abs_shap_value", ascending=False)
                logger.info("\n" + str(shap_summary_df_0.head(20)))  # Log top 20 features

            else:
                # Use KernelExplainer for other models
                background = shap.sample(X_train_scaled, 100)  # Use a small background dataset
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_val_sample)

                # Handle binary classification
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    # SHAP summary for class 1 (positive class)
                    logger.info(f"SHAP summary for class 1 (positive class):")
                    shap_summary_class_1 = np.mean(np.abs(shap_values[1]), axis=0)
                    shap_summary_df_1 = pd.DataFrame({
                        "feature": feature_cols,
                        "mean_abs_shap_value": shap_summary_class_1
                    }).sort_values("mean_abs_shap_value", ascending=False)
                    logger.info("\n" + str(shap_summary_df_1.head(20)))  # Log top 20 features

                    # SHAP summary for class 0 (negative class)
                    logger.info(f"SHAP summary for class 0 (negative class):")
                    shap_summary_class_0 = np.mean(np.abs(shap_values[0]), axis=0)
                    shap_summary_df_0 = pd.DataFrame({
                        "feature": feature_cols,
                        "mean_abs_shap_value": shap_summary_class_0
                    }).sort_values("mean_abs_shap_value", ascending=False)
                    logger.info("\n" + str(shap_summary_df_0.head(20)))  # Log top 20 features

            logger.info(f"SHAP analysis completed for {model_name}.")
        except Exception as e:
            logger.error(f"Error during SHAP analysis for {model_name}: {str(e)}")
    else:
        logger.info(f"SHAP analysis not supported for {model_name}.")

if __name__ == "__main__":

    logger.info("Loading environment variables...")
    customer_purchase_labels_path = os.environ.get('customer_purchase_labels_path')
    gcs_embeddings_output_path = os.environ.get('gcs_embeddings_output_path')
    last_week_emb_mapping_table = os.environ.get('last_week_emb_mapping_table')
    predictions_output_table = os.environ.get('predictions_output_table')
    customer_purchase_profile_path = os.environ.get('customer_purchase_profile_path')
    customer_purchase_history_path = os.environ.get('customer_purchase_history_path')
    customer_point_profile_path = os.environ.get('customer_point_profile_path')
    npp_history_path = os.environ.get('npp_history_path')
    last_week_identifier = os.environ.get('last_week_identifier')
    statistic_features_path = os.environ.get('statistic_features_path')
    model_output_path = os.environ.get('model_output_path')
    model_features_path = os.environ.get('model_features_path')


    logger.info("Starting bigquery client and GCS file system...")
    bq_project = os.environ.get('bq_project')
    bq_client = bigquery.Client(project=bq_project)
    fs               = gcsfs.GCSFileSystem()

    reco_reference_pd = load_reco_reference_df(bq_client=bq_client)
    
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

    #Customer Purchase Profile data
    logger.info(f"Loading customer purchase profile data from: {customer_purchase_profile_path}")
    customer_purchase_profile = [_path for _path in fs.glob(os.path.join(customer_purchase_profile_path, "*.parquet"))]
    customer_purchase_profile_ds = parquet.ParquetDataset(customer_purchase_profile, filesystem=fs)
    customer_purchase_profile_pd_df = customer_purchase_profile_ds.read().to_pandas()
    customer_purchase_profile_pd_df = customer_purchase_profile_pd_df[customer_purchase_profile_pd_df['banner'] == 'CTR']   
    customer_purchase_profile_pd_df.drop(["epcl_profileid", "banner"], axis=1, inplace=True, errors='ignore')
    fillna_median_cols = ["days_since_last_purchase_p25", "days_since_last_purchase_p50", "days_since_last_purchase_p75", 
                           "avg_days_since_last_purchase", "stddev_days_since_last_purchase"]
    customer_purchase_profile_pd_df[fillna_median_cols] = customer_purchase_profile_pd_df[fillna_median_cols].fillna(customer_purchase_profile_pd_df[fillna_median_cols].median())
    logger.info(f"Loaded {len(customer_purchase_profile_pd_df)} customer purchase profiles.")

    # Customer Purchase History data
    logger.info(f"Loading customer purchase history data from: {customer_purchase_history_path}")
    customer_purchase_history = [_path for _path in fs.glob(os.path.join(customer_purchase_history_path, "*.parquet"))]
    customer_history_ds = parquet.ParquetDataset(customer_purchase_history, filesystem=fs)
    customer_history_pd_df = customer_history_ds.read().to_pandas()
    customer_history_pd_df = customer_history_pd_df[customer_history_pd_df['banner'] == 'CTR']
    customer_history_pd_df.drop(["epcl_profileid", "banner", last_week_identifier], axis=1, inplace=True, errors='ignore')
    logger.info(f"Loaded {len(customer_history_pd_df)} customer purchase histories.")
    customer_history_pivoted_df = customer_history_pd_df.pivot_table(
                                            index='epsilon_id',
                                            columns='week_identifier',
                                            values='made_purchase',
                                            aggfunc='max'  # or 'sum' if multiple purchases should be counted
                                        )\
                                .reset_index() \
                                .fillna(0.0)
    customer_history_final_df = rename_pivot_columns(customer_history_pivoted_df, 
                                                     reco_reference_pd, 
                                                     last_week_identifier,
                                                     "_weeks_back_purchase", 
                                                     removal_cols=["epsilon_id"])
    
    #Customer Deal Season data
    logger.info(f"Loading customer deal season data from bigquery table...")
    last_year_week_identifier = str(int(last_week_identifier[0:4]) - 1) + last_week_identifier[4:]
    reco_reference_filtered = reco_reference_pd[reco_reference_pd["week_identifier"] <= last_year_week_identifier]
    customer_deal_season_pd_df = customer_history_pd_df.merge(reco_reference_pd, on='week_identifier', how='inner') \
                          .pivot_table(
                                            index='epsilon_id',
                                            columns='deal_season_num',
                                            values='made_purchase',
                                            aggfunc='max'
                                        )\
                                .reset_index() \
                                .fillna(0)
    
    customer_ids_deal_season_pd_df = customer_deal_season_pd_df["epsilon_id"]
    deal_season_data_pd_df = customer_deal_season_pd_df[customer_deal_season_pd_df.columns.drop("epsilon_id")]
    int_columns = {col: int(col) for col in deal_season_data_pd_df.columns}
    deal_season_data_pd_adjusted = deal_season_data_pd_df.rename(columns=int_columns).add_prefix("deal_season_num_")
    customer_deal_season_final_pd_df = pd.concat([customer_ids_deal_season_pd_df, deal_season_data_pd_adjusted], axis=1)
    logger.info(f"Loaded {len(customer_deal_season_final_pd_df)} customer deal season profiles.")


    # Customer Point Profile data
    logger.info(f"Loading customer point profile data from: {customer_point_profile_path}")
    customer_point_profile = [_path for _path in fs.glob(os.path.join(customer_point_profile_path, "*.parquet"))]
    customer_point_profile_ds = parquet.ParquetDataset(customer_point_profile, filesystem=fs)
    customer_point_profile_pd_df = customer_point_profile_ds.read().to_pandas()
    customer_point_profile_pd_df = customer_point_profile_pd_df[customer_point_profile_pd_df['banner'] == 'CTR']
    customer_point_profile_pd_df.drop("banner", axis=1, inplace=True, errors='ignore')
    logger.info(f"Loaded {len(customer_point_profile_pd_df)} customer point profiles.")
    

    customer_profile_summary_df = customer_point_profile_pd_df.merge(customer_purchase_profile_pd_df, on='epsilon_id', how='left')\
                                                              .merge(customer_history_final_df, on='epsilon_id', how='left') \
                                                              .merge(customer_deal_season_final_pd_df, on='epsilon_id', how='left')
    
    customer_profile_summary_df.fillna(0, inplace=True)


    # NPP data
    logger.info(f"Loading NPP data from : {npp_history_path}")
    npp_history = [_path for _path in fs.glob(os.path.join(npp_history_path, "*.parquet"))]
    npp_history_ds = parquet.ParquetDataset(npp_history, filesystem=fs)
    npp_history_pd_df = npp_history_ds.read().to_pandas()
    logger.info(f"Loaded {len(npp_history_pd_df)} NPP history records.")
    npp_history_pd_df = npp_history_pd_df[npp_history_pd_df["week_identifier"] <=last_week_identifier]
    npp_history_pivoted_df = npp_history_pd_df.set_index('week_identifier')['npp_flag'].T.to_frame().T
    npp_history_pivoted_df.reset_index(drop=True, inplace=True)
    npp_history_pivoted_df = rename_pivot_columns(npp_history_pivoted_df, reco_reference_pd, last_week_identifier, "_weeks_back_NPP")
    npp_history_pivoted_df = npp_history_pivoted_df.add_suffix('_npp_flag')

    # Statistic Features
    logger.info(f"Loading statistic features from: {statistic_features_path}")
    statistic_features = [_path for _path in fs.glob(os.path.join(statistic_features_path, "*.parquet"))]
    statistic_features_ds = parquet.ParquetDataset(statistic_features, filesystem=fs)
    statistic_features_pd_df = statistic_features_ds.read().to_pandas()
    statistic_features_pd_df.drop(["epcl_profileid", "dispersion"], axis=1, inplace=True, errors='ignore')
    logger.info(f"Loaded {len(statistic_features_pd_df)} statistic features records.")



    # Combine Model Features DataFrame
    model_features_df = embedding_df.merge(customer_profile_summary_df, on='epsilon_id', how='left') \
                                    .merge(statistic_features_pd_df, on='epsilon_id', how='left') \
                                    .merge(npp_history_pivoted_df, how='cross') \
                                    .merge(customer_labels_pd_df[['epsilon_id', 'purchase_flag']], on='epsilon_id', how='left') \

    model_features_df["purchase_flag"] = model_features_df["purchase_flag"].fillna('N')
    model_features_df["prob"] = model_features_df["prob"].fillna(model_features_df["prob"].median())
    model_features_df["aided_prob"] = model_features_df["aided_prob"].fillna(model_features_df["aided_prob"].median())

    logger.info("Model features shape: ", model_features_df.shape)
    logger.info(model_features_df.head())
    logger.info(model_features_df.groupby('purchase_flag').count())
    
    # =============================================
    # Prepare data for training
    # =============================================
    #  # Get feature columns (latent_0 to latent_199)
    # feature_cols = [f'latent_{i}' for i in range(200)]
    
    # # Verify all feature columns exist
    # if not all(col in model_features_df.columns for col in feature_cols):
    #     missing_cols = [col for col in feature_cols if col not in model_features_df.columns]
    #     logger.error(f"Missing feature columns: {missing_cols}")
    #     raise ValueError("Missing expected feature columns")

    feature_cols = model_features_df.columns.drop(["epsilon_id", "epcl_profileid", "lightfm_id", "user_id", "purchase_flag", "enroll_date", "register_date", "first_purchase_date", "last_purchase_date"]).tolist()
    logger.info(f"Feature columns used for training: {feature_cols}")

    # Prepare X (features) and y (target)
    X = model_features_df[feature_cols]
    y = (model_features_df['purchase_flag'] == 'Y').astype(int)  # Convert to binary

    # Split data first
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    # Detect binary and numeric columns
    column_vals = ["weeks_back", "deal_season"]
    binary_cols = []
    for column in X_train.columns:
        if any (val in column for val in column_vals):
            binary_cols.append(column)

    numeric_cols = [col for col in X_train.columns if col not in binary_cols]
    logger.info(f"Numeric columns: {numeric_cols}")
    logger.info(f"Binary columns: {binary_cols}")



    # Fit scaler only on training data
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('bin', 'passthrough', binary_cols)
    ])

    # Scale the training data
    X_train_scaled = preprocessor.fit_transform(X_train)

    # Scale validation and test data using the same parameters
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Validation set size: {len(X_val)}")
    logger.info(f"Test set size: {len(X_test)}")

    
    
    if hyperparameters_tuning:
        best_params = hyperparameter_optimization(X, y)
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
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        ),
        'LightGBM': LGBMClassifier(
            # **best_params,
            n_estimators=500,
            max_depth=-1,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=42
        ),
        # 'DeepLearning': KerasClassifierWrapper(
        #     input_dim=X_train_scaled.shape[1],
        #     scale_pos_weight=scale_pos_weight
        #     # learning_rate=0.0005  
        # )
    }

    # Evaluate all models
    results = {}
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        results[model_name] = evaluate_classifier(
            model, model_name, 
            X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test
    )

    # Feature importance analysis
    for model_name, model in models.items():
        logger.info(f"\nFeature importance for {model_name}:")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": importances
            }).sort_values("importance", ascending=False)
            logger.info("\n" + str(importance_df.head(40)))  # Show top 20
        elif model_name == "DeepLearning":
            # Use permutation importance for deep learning model
            try:
                logger.info("Calculating permutation importance for DeepLearning model (may be slow)...")
                result = permutation_importance(
                    model, X_val_scaled, y_val, n_repeats=5, random_state=42, scoring='roc_auc',  response_method="predict"
                )
                importance_df = pd.DataFrame({
                    "feature": feature_cols,
                    "importance": result.importances_mean
                }).sort_values("importance", ascending=False)
                logger.info("\n" + str(importance_df.head(40)))
            except Exception as e:
                logger.error(f"Error calculating permutation importance for DeepLearning model: {str(e)}")
                logger.info("Skipping feature importance for DeepLearning model due to error.")
        else:
            logger.info("Feature importance not available for this model.")

    # Perform SHAP analysis for each model
    for model_name, model in models.items():
        shap_analysis(model, model_name, X_train_scaled, X_val_scaled, feature_cols)

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


    batch_upload_to_gcs(df=X, 
                        output_path=model_features_path,
                        batch_size=400000,
                        file_name=f'model_features_{last_week_identifier}',
                        fs=fs)


    # # Generate predictions for all rows in model_features_df
    # X_scaled = preprocessor.transform(X)  # Scale the full feature set using the preprocessor fitted on X_train
    # model_features_df['predicted_purchase_prob'] = best_model.predict_proba(X_scaled)[:, 1]
    
    # batch_size = 400000  
    # # Save predictions with customer IDs
    # predictions_df = model_features_df[['epsilon_id', 'lightfm_id', 'user_id', 'predicted_purchase_prob']]
    # batch_upload(
    #     batch_size=batch_size, 
    #     df=predictions_df, 
    #     table_name=predictions_output_table, 
    #     source_format=bigquery.SourceFormat.PARQUET, 
    #     bq_client=bq_client
    # )
    
    # logger.info(f"Predictions saved to table: {predictions_output_table}")
    # logger.info("Model comparison and evaluation completed successfully.")