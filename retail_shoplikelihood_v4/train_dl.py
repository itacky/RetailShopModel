import logging
import sys
import os
from google.cloud import bigquery
from pyarrow import parquet
import gcsfs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import numpy as np
from joblib import dump
import optuna
from optuna.pruners import HyperbandPruner
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
from pythonjsonlogger import jsonlogger


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    logger.info("GPU memory growth enabled.")
else:
    logger.warning("No GPU devices found. Using CPU.")

hyperparameters_tuning = True 
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

def null_identifier(df, df_name):
    """Identify null values in the DataFrame."""
    logger.info(f"Checking for null values in {df_name} DataFrame...")
    null_counts = df.isnull().sum()
    null_summary = null_counts[null_counts > 0]
    if null_summary.empty:
        logger.info("No null values found in DataFrame.")
    else:
        logger.info(f"Null values in DataFrame:\n{null_summary}")

def find_optimal_threshold(y_true, y_probs):
    prec, rec, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-9)  # Avoid division by zero

    #find the threshold that gives the maximum F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    logger.info(f"Optimal Threshold: {optimal_threshold:.4f} with F1 Score: {optimal_f1:.4f}")
    return optimal_threshold, optimal_f1


def log_performance_metrics_to_bq(bq_client, df, table_name):
        job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET, write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
        job = bq_client.load_table_from_dataframe(df, table_name, job_config=job_config)
        job.result()


class TensorFlowClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, batch_size=1024, epochs=50, learning_rate=0.001, patience=5, random_state=42):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.random_state = random_state
        self.model = None

    def _build_model(self):
        """Build the TensorFlow model."""
        model = Sequential([
            Dense(1024, activation='relu', input_dim=self.input_dim),
            BatchNormalization(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def fit(self, X, y, validation_data=None):
        """Train the TensorFlow model."""
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        self.model = self._build_model()
        early_stopping = EarlyStopping(
            monitor='loss', patience=self.patience, restore_best_weights=True
        )
        self.model.fit(
            X, y,
            validation_data=validation_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[early_stopping],
            verbose=1
        )
        return self

    def predict(self, X):
        """Predict class labels."""
        proba = self.model.predict(X).flatten()
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        """Predict class probabilities."""
        proba = self.model.predict(X).flatten()
        return np.vstack([1 - proba, proba]).T
    

def create_tf_model(trial, input_dim):
    """Create a TensorFlow model with hyperparameters suggested by Optuna."""
    model = Sequential()

    # Number of layers and units per layer
    num_layers = trial.suggest_int("num_layers", 1, 4)
    for i in range(num_layers):
        units = trial.suggest_int(f"units_layer_{i}", 64, 1024, step=64)
        if i == 0:
            #Input layer
            model.add(Dense(units, activation="relu", input_dim=input_dim, kernel_regularizer=l2(1e-4)))
        else:
            model.add(Dense(units, activation="relu", kernel_regularizer=l2(1e-4)))
        model.add(BatchNormalization())
        dropout_rate = trial.suggest_float(f"dropout_layer_{i}", 0.0, 0.5)
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation="sigmoid"))

    # Optimizer and learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def objective_with_cv(trial, X, y):
    """Objective function with cross-validation."""
    input_dim = X.shape[1]
    auc_scores = []

    # 5-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create and train the model
        model = create_tf_model(trial, input_dim)
        early_stopping = EarlyStopping(
            monitor="val_auc", patience=5, restore_best_weights=True
        )
        lr_schedule = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        )
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=trial.suggest_int("batch_size", 32, 2048, step=64),
            epochs=50,
            callbacks=[early_stopping, lr_schedule],
            verbose=0
        )

        # Evaluate the model
        y_val_pred = model.predict(X_val).flatten()
        auc = roc_auc_score(y_val, y_val_pred)
        auc_scores.append(auc)

    return np.mean(auc_scores)

def tune_hyperparameters_with_optuna(X, y, n_trials=50):
    """Run Optuna hyperparameter tuning with cross-validation and regularization."""
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True),
                                storage="sqlite:///optuna_study.db",
                                study_name="tf_cpu_opt",
                                load_if_exists=True,
                                pruner=HyperbandPruner(min_resource=50, max_resource=5000, reduction_factor=3))
    study.optimize(lambda trial: objective_with_cv(trial, X, y), 
                   n_trials=n_trials,
                   n_jobs=-1,
                   gc_after_trial=True,
                   show_progress_bar=True)

    logger.info("Best trial:")
    logger.info(f"  Value: {study.best_trial.value}")
    logger.info("  Params: ")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")

    return study


if __name__ == "__main__":
    os.environ['PYTHONVERBOSE'] = '1'
    logger.info("Loading environment variables...")
    customer_purchase_labels_path = os.environ.get('customer_purchase_labels_path')
    gcs_embeddings_output_path = os.environ.get('gcs_embeddings_output_path')
    predictions_output_table = os.environ.get('predictions_output_table')
    customer_purchase_profile_path = os.environ.get('customer_purchase_profile_path')
    customer_purchase_history_path = os.environ.get('customer_purchase_history_path')
    customer_point_profile_path = os.environ.get('customer_point_profile_path')
    npp_history_path = os.environ.get('npp_history_path')
    week_identifier = os.environ.get('week_identifier')
    target_week_identifier = os.environ.get('target_week_identifier')
    current_week_identifier = os.environ.get('current_week_identifier')
    statistic_features_path = os.environ.get('statistic_features_path')
    model_output_path = os.environ.get('model_output_path')
    model_features_path = os.environ.get('model_features_path')
    model_performance_output_table = os.environ.get('model_performance_output_table')


    logger.info("Starting bigquery client and GCS file system...")
    bq_project = os.environ.get('bq_project')
    bq_client = bigquery.Client(project=bq_project)
    fs               = gcsfs.GCSFileSystem()

    reco_reference_pd = load_reco_reference_df(bq_client=bq_client)
    
    # Customer purchase labels
    logger.info(f"Loading customer purchase labels from: {customer_purchase_labels_path}")
    customer_purchase_labels = [_path for _path in fs.glob(os.path.join(customer_purchase_labels_path, "*.parquet"))]
    customer_labels_ds       = parquet.ParquetDataset(customer_purchase_labels, filesystem=fs)
    customer_labels_pd_df    = customer_labels_ds.read().to_pandas()
    logger.info(f"Loaded {len(customer_labels_pd_df)} customer purchase labels.")

    # Embedding mapping table
    logger.info(f"Loading LightFM embeddings from: {gcs_embeddings_output_path}")
    embeddings_files = [_path for _path in fs.glob(os.path.join(gcs_embeddings_output_path, "*.parquet"))]
    embedding_ds = parquet.ParquetDataset(embeddings_files, filesystem=fs)
    embedding_df = embedding_ds.read().to_pandas()
    logger.info(f"Total embedding rows loaded: {len(embedding_df)}")
    logger.info(f"Embedding columns: {embedding_df.columns}")

    #Customer Purchase Profile data
    logger.info(f"Loading customer purchase profile data from: {customer_purchase_profile_path}")
    customer_purchase_profile = [_path for _path in fs.glob(os.path.join(customer_purchase_profile_path, "*.parquet"))]
    customer_purchase_profile_ds = parquet.ParquetDataset(customer_purchase_profile, filesystem=fs)
    customer_purchase_profile_pd_df = customer_purchase_profile_ds.read().to_pandas()
    logger.info(f"Purchase profile columns: {customer_purchase_profile_pd_df.columns}")
    logger.info(f"Loaded {len(customer_purchase_profile_pd_df)} customer purchase profiles.")
    customer_purchase_profile_pd_df = customer_purchase_profile_pd_df[customer_purchase_profile_pd_df['banner'] == 'CTR']   
    customer_purchase_profile_pd_df.drop(["epcl_profileid", "banner"], axis=1, inplace=True, errors='ignore')
    fillna_median_cols = ["days_since_last_purchase_p25", "days_since_last_purchase_p50", "days_since_last_purchase_p75", 
                           "avg_days_since_last_purchase", "stddev_days_since_last_purchase"]
    customer_purchase_profile_pd_df[fillna_median_cols] = customer_purchase_profile_pd_df[fillna_median_cols].fillna(customer_purchase_profile_pd_df[fillna_median_cols].median())
   

    # Customer Purchase History data
    logger.info(f"Loading customer purchase history data from: {customer_purchase_history_path}")
    customer_purchase_history = [_path for _path in fs.glob(os.path.join(customer_purchase_history_path, "*.parquet"))]
    customer_history_ds = parquet.ParquetDataset(customer_purchase_history, filesystem=fs)
    customer_history_pd_df = customer_history_ds.read().to_pandas()
    customer_history_pd_df = customer_history_pd_df[customer_history_pd_df['banner'] == 'CTR']
    customer_history_pd_df.drop(["epcl_profileid", "banner", week_identifier], axis=1, inplace=True, errors='ignore')
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
                                                     week_identifier,
                                                     "_weeks_back_purchase", 
                                                     removal_cols=["epsilon_id"])
    
    #Customer Deal Season data
    logger.info(f"Loading customer deal season data from bigquery table...")
    last_year_week_identifier = str(int(week_identifier[0:4]) - 1) + week_identifier[4:]
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
    
    new_cols = []
    for c in customer_deal_season_pd_df.columns:
        if c == "epsilon_id":
            new_cols.append(c)
        else:
            new_cols.append(f"deal_season_num_{int(c)}")
    customer_deal_season_final_pd_df = customer_deal_season_pd_df.copy()
    customer_deal_season_final_pd_df.columns = new_cols
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
    npp_history_pd_df = npp_history_pd_df[npp_history_pd_df["week_identifier"] <=week_identifier]
    npp_history_pivoted_df = npp_history_pd_df.set_index('week_identifier')['npp_flag'].T.to_frame().T
    npp_history_pivoted_df.reset_index(drop=True, inplace=True)
    npp_history_pivoted_df = rename_pivot_columns(npp_history_pivoted_df, reco_reference_pd, week_identifier, "_weeks_back_NPP")
    npp_history_pivoted_df = npp_history_pivoted_df.add_suffix('_npp_flag')




    # Combine Model Features DataFrame
    logger.info(embedding_df.columns.tolist())
    logger.info(customer_profile_summary_df.columns.tolist())
    model_features_df = embedding_df.merge(customer_profile_summary_df, on='epsilon_id', how='inner') \
                                    .merge(npp_history_pivoted_df, how='cross') \
                                    .merge(customer_labels_pd_df[['epsilon_id', 'purchase_flag']], on='epsilon_id', how='left') \
                                    .drop_duplicates(subset='epsilon_id')

    model_features_df["purchase_flag"] = model_features_df["purchase_flag"].fillna('N')
    
    #Check feature DataFrames for null values
    null_identifier(embedding_df, "Embedding DataFrame")
    null_identifier(customer_profile_summary_df, "Customer Profile Summary DataFrame")
    null_identifier(npp_history_pivoted_df, "NPP History DataFrame")
    null_identifier(model_features_df, "Model Features DataFrame")

    #Check merged DataFrames for null values
    null_identifier(embedding_df.merge(customer_profile_summary_df, on='epsilon_id', how='inner'), "First Merge DataFrame")
    null_identifier(embedding_df.merge(customer_profile_summary_df, on='epsilon_id', how='inner') \
                                    .merge(npp_history_pivoted_df, how='cross'), "Second Merge DataFrame")


    logger.info(f"Model features shape: {model_features_df.shape}")
    logger.info(model_features_df.head())
    logger.info(model_features_df.groupby('purchase_flag').count())
    
    # Define the segments
    segment_1 = model_features_df[model_features_df['days_since_last_purchase'] <= 365]
    segment_2 = model_features_df[model_features_df['days_since_last_purchase'] <= 730]
    segment_3 = model_features_df

    
    # =============================================
    # Prepare data for training
    # =============================================
    feature_cols = model_features_df.columns.drop(["epsilon_id", "epcl_profileid", "lightfm_id", "user_id", "purchase_flag", "enroll_date", "register_date", "first_purchase_date", "last_purchase_date"], errors='ignore').tolist()

    
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


    # Calculate class weight for XGBoost and LightGBM
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    default_params = {
    "n_estimators":500,
    "max_depth":8,
    "learning_rate":0.05,
    "subsample":0.8,
    "colsample_bytree":0.8,
    "scale_pos_weight":scale_pos_weight,
    "tree_method":'hist',
    "n_jobs":-1,
    "random_state":42
    }   
    best_params = default_params

    # Initialize all models
    models = {
        'tensorflow': TensorFlowClassifier(
        input_dim=X_train_scaled.shape[1],
        batch_size=2048,
        epochs=50,
        learning_rate=0.003,
        patience=5,
        random_state=42
    
    ),
    }

    
    if hyperparameters_tuning:
        logger.info("Hyperparameter tuning is enabled. Starting optimization...")
        study = tune_hyperparameters_with_optuna(X_train_scaled, y_train, n_trials=30)

        # Use the best hyperparameters to create the final model
        best_params = study.best_trial.params
        logger.info(f"Best hyperparameters: {best_params}")

        # Create the final TensorFlow model with the best hyperparameters
        final_model = create_tf_model(study.best_trial, input_dim=X_train_scaled.shape[1])
    else:
        logger.info("Hyperparameter tuning is disabled. Using default parameters.")
        final_model = TensorFlowClassifier(
            input_dim=X_train_scaled.shape[1],
            batch_size=2048,
            epochs=50,
            learning_rate=0.003,
            patience=5,
            random_state=42
        )
    
    results = {}
    model = final_model
    model_name = "tensorflow_optimized"
    logger.info(f"\nEvaluating {model_name}...")
    results[model_name] = evaluate_classifier(
        model, model_name, 
        X_train_scaled, X_val_scaled, X_test_scaled, 
        y_train, y_val, y_test
    )


    # Select best model based on validation AUC-ROC
    best_model_name = max(results.items(), key=lambda x: x[1]['val_auc'])[0]
    logger.info(f"Best performing model: {best_model_name}")

    #Create predictions output table in bigquery
    test_predictions = results[best_model_name]['test_predictions']
    test_probabilities = results[best_model_name]['test_probabilities']
    classification_report_dict = classification_report(y_test.reset_index(drop=True), test_predictions.reset_index(drop=True), output_dict=True)
    roc_auc = roc_auc_score(y_test, test_probabilities)

    # Find optimal threshold based on F1 score
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, test_probabilities)

    # Extract metrics for the classification summary table
    classification_summary = {
        'model_type': best_model_name,
        'banner': 'CTR',
        'roc_auc': roc_auc,
        'precision_0': classification_report_dict.get('0', {}).get('precision', None),
        'recall_0': classification_report_dict.get('0', {}).get('recall', None),
        'f1_0': classification_report_dict.get('0', {}).get('f1-score', None),
        'support_0': classification_report_dict.get('0', {}).get('support', None),
        'precision_1': classification_report_dict.get('1', {}).get('precision', None),
        'recall_1': classification_report_dict.get('1', {}).get('recall', None),
        'f1_1': classification_report_dict.get('1', {}).get('f1-score', None),
        'support_1': classification_report_dict.get('1', {}).get('support', None),
        'accuracy': classification_report_dict.get('accuracy', None),
        'support_total': classification_report_dict.get('macro avg', {}).get('support', None),
        'macro_f1': classification_report_dict.get('macro avg', {}).get('f1-score', None),
        'weighted_f1': classification_report_dict.get('weighted avg', {}).get('f1-score', None),
        'train_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'feature_cols': feature_cols,
        'hyperparameters': json.dumps(best_params, ensure_ascii=False),
        'optimal_threshold': optimal_threshold,
        'optimal_f1': optimal_f1,
        'top_20_shap_features_class_1': None,
        'top_20_shap_features_class_0': None,
        'week_identifier': current_week_identifier,
    }

    
    #Save classification summary to bigquery
    classification_summary_df = pd.DataFrame([classification_summary])
    classification_summary_df = classification_summary_df.fillna(0)
    log_performance_metrics_to_bq(bq_client=bq_client, 
                                  df=classification_summary_df, 
                                  table_name=model_performance_output_table)
    
    # Save test predictions to bigquery
    predictions_df = results[best_model_name]['results_table'].reset_index()
    predictions_df.rename(columns={'index': 'epsilon_id'}, inplace=True)
    try:
        batch_upload(batch_size=400000, 
                     df=predictions_df, 
                     table_name=predictions_output_table, 
                     source_format=bigquery.SourceFormat.PARQUET,
                     bq_client=bq_client)
    except Exception as e:
        logger.error(f"Failed to upload predictions to BigQuery: {e}")
        raise
