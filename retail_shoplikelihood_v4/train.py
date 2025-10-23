import logging
import sys
import os
from google.cloud import bigquery
from pyarrow import parquet
import gcsfs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

import json
import numpy as np
from joblib import dump
import time
import optuna
from functools import partial
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import shap
from scipy.special import expit  # Sigmoid function
from datetime import datetime


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


logger = logging.getLogger(__name__)

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
        job.result()
        print(f"Uploaded rows {start} to {end} to {table_name}")
    print("All batches uploaded successfully.")


def load_from_bigquery(query, bq_client):
    return bq_client.query(query).to_dataframe()


def load_reco_reference_df(bq_client):
    logger.info("Loading deal week reference table...")
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


def load_triangle_lifecycle_segment_df(bq_client, tlc_date, segment):
    logger.info("Loading triangle lifecycle segment table...")
    query = f""" 
    SELECT 
        epsilon_id
    FROM `dp-lore.loyalty_reco.customer_lifecycle`
    WHERE dateid= '{tlc_date}'
    and status = '{segment}'
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


# Define the objective function
def objective(trial, X, y):
    """
    Optuna objective for XGBoost on CPU (Vertex sklearn-cpu image).
    - Uses hist trees, high-leverage knobs (min_child_weight, max_leaves, etc.)
    - Optimizes PR-AUC (aucpr) which is better under class imbalance
    - Early stopping with large n_estimators so ES picks the round count
    - Coordinates per-trial threads to play nicely with parallel Optuna workers
    """
 
    # Cast once for speed/memory
    try:
        X = X.astype(np.float32, copy=False)
    except Exception:
        pass

    # Compute once (don’t re-count every param change)
    y_arr = np.asarray(y)
    pos = np.sum(y_arr == 1)
    neg = np.sum(y_arr == 0)
    spw = float(neg / max(1, pos))

    # -------- Search space (CPU-optimized) --------
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "booster": "gbtree",
        "random_state": 42,

        # Coordinate with study.optimize(..., n_jobs=K)
        "n_jobs": trial.suggest_categorical("xgb_n_jobs", [4, 8, 16]),

        # Learning & structure
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "grow_policy": trial.suggest_categorical("grow_policy", ["lossguide", "depthwise"]),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "max_leaves": trial.suggest_int("max_leaves", 63, 1023, log=True),

        # Regularization
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-1, 256.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 50.0, log=True),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),

        # Row/feature sampling
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.4, 1.0),

        # Histogram controls
        "max_bin": trial.suggest_int("max_bin", 64, 256, log=True),

        # Imbalance
        "scale_pos_weight": spw,
    }

    # Use a reasonable validation size for ES stability
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    clf = XGBClassifier(
        **params,
        early_stopping_rounds=100 
    )

    # Train with early stopping + pruning on aucpr
    clf.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    verbose=False
    
)

    # Optimize PR-AUC; (you still log ROC elsewhere in your pipeline)
    preds = clf.predict_proba(X_va)[:, 1]
    return roc_auc_score(y_va, preds)


def hyperparameter_optimization(X, y):
    """
    Stage-1 Optuna search on a stratified 100k sample (as you had),
    with safe parallelism and a strong pruner. Returns best params.
    """
    import numpy as np
    import optuna
    from sklearn.model_selection import train_test_split
    from optuna.pruners import HyperbandPruner

    logger.info("Starting hyperparameter optimization with Optuna...")

    # Stratified sample for faster search
    sample_size = 250000/len(y) if len(y) > 250000 else 0.8
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=sample_size, random_state=42, stratify=y
    )

    # Study with TPE + Hyperband pruning for early exits
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True, group=True),
        storage="sqlite:///optuna_study.db",
        study_name="xgb_cpu_opt",
        load_if_exists=True,
        pruner=HyperbandPruner(min_resource=50, max_resource=5000, reduction_factor=3)
    )

    # IMPORTANT: avoid oversubscription; set a modest parallelism
    # Tune this so: (n_jobs * median xgb_n_jobs) ~= available vCPUs
    n_trials = 100  
    study.optimize(
        partial(objective, X=X_sample, y=y_sample),
        n_trials=n_trials,
        n_jobs=4,
        gc_after_trial=True,
        show_progress_bar=True
    )

    logger.info("Best trial:")
    logger.info(f"  Value (AUC-PR): {study.best_trial.value:.6f}")
    logger.info("  Params:")
    for k, v in study.best_trial.params.items():
        logger.info(f"    {k}: {v}")

    # Return only XGB params 
    best = dict(study.best_trial.params)
    best.pop("xgb_n_jobs", None) # Remove n_jobs to let final model use all cores
    return best


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
        'test_predictions': pd.Series(test_predictions, index=y_test.index),
        'val_probabilities': val_prob_predictions,
        'test_probabilities':  pd.Series(test_prob_predictions, index=y_test.index),
        'results_table': pd.DataFrame({
            'actual': y_test,
            'predicted': test_predictions,
            'predicted_proba': test_prob_predictions
        }, index=y_test.index)
    }


def train_final_model(model, X, y):
    
    """Train the final model on the entire dataset."""
    start = time.time()
    logger.info("Training final model on the entire dataset...")
    model.fit(X, y)
    end = time.time()
    logger.info(f"Final model training completed in {end - start:.2f} seconds.")
    return model


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
            if model_name in ["randomforest", "xgboost", "lightgbm"]:
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
                return shap_summary_df_1, shap_summary_df_0

            else:
                logger.info(f"SHAP analysis not implemented for model type: {model_name}")
                logger.info("Skipping SHAP analysis.")

            logger.info(f"SHAP analysis completed for {model_name}.")
        except Exception as e:
            logger.error(f"Error during SHAP analysis for {model_name}: {str(e)}")
    else:
        logger.info(f"SHAP analysis not supported for {model_name}.")


def evaluate_segment(model, segment_df, feature_cols, target_col='purchase_flag'):
    """Evaluate the model for a specific segment."""
    X_segment = segment_df[feature_cols]
    y_segment = (segment_df[target_col] == 'Y').astype(int)  # Convert to binary

    # Scale the features using the preprocessor
    X_segment_scaled = preprocessor.transform(X_segment)

    # Evaluate the model using the evaluate_classifier function
    segment_results = evaluate_classifier(
        model=model,
        model_name="Segment Evaluation",
        X_train=X_segment_scaled,
        X_val=X_segment_scaled,  # Use the same data for validation
        X_test=X_segment_scaled,  # Use the same data for testing
        y_train=y_segment,
        y_val=y_segment,
        y_test=y_segment
    )

    return segment_results


def  null_identifier(df, df_name):
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


if __name__ == "__main__":

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
    tlc_segment = 'Onboarding'


    logger.info("Starting bigquery client and GCS file system...")
    bq_project = os.environ.get('bq_project')
    bq_client = bigquery.Client(project=bq_project)
    fs               = gcsfs.GCSFileSystem()

    reco_reference_pd = load_reco_reference_df(bq_client=bq_client)
    
    #triangle_lifecycle_segment data
    tlc_end_date = reco_reference_pd[reco_reference_pd["week_identifier"] == week_identifier].end_date.iloc[0]
    tlc_segments_df = load_triangle_lifecycle_segment_df(bq_client=bq_client, tlc_date=tlc_end_date, segment=tlc_segment)
    logger.info(f"Loaded {len(tlc_segments_df)} triangle lifecycle segment records for segment '{tlc_segment}'. Date: {tlc_end_date}")

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
    feature_cols = model_features_df.columns.drop(["epsilon_id", "epcl_profileid", "lightfm_id", "user_id", "purchase_flag", "enroll_date", "register_date", "first_purchase_date", "last_purchase_date"]).tolist()
    logger.info(f"Feature columns used for training: {feature_cols}")

    # Prepare X (features) and y (target)
    X = model_features_df[feature_cols]
    y = (model_features_df['purchase_flag'] == 'Y').astype(int)  # Convert to binary

    # Split data
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


    # Calculate class weight for XGBoost and LightGBM
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"Class distribution in training set: {y_train.value_counts().to_dict()}")
    logger.info(f"Scale pos weight for handling class imbalance: {scale_pos_weight}")

    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    # Initialize all models
    default_params = {
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": 'hist',
        "n_jobs": -1,
        "random_state": 42
    }
    best_params = default_params

    models = {
        'xgboost': XGBClassifier(**best_params),
        'lightgbm': LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        ),
        'catboost': CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            subsample=0.8,
            scale_pos_weight=scale_pos_weight,
            random_seed=42,
            verbose=0  # Suppress CatBoost logs
        ),
        'adaboost': AdaBoostClassifier(
            base_estimator=DecisionTreeClassifier(
                max_depth=1,  # Default weak learner
                class_weight="balanced"  # Handle class imbalance
            ),
            n_estimators=500,
            learning_rate=0.05,
            random_state=42
        )
    }

    # Perform hyperparameter optimization if enabled
    if hyperparameters_tuning:
        logger.info("Hyperparameter tuning is enabled. Starting optimization...")
        best_params = hyperparameter_optimization(X_train, y_train)
        logger.info(f"Best hyperparameters found: {best_params}")

        # Update the XGBoost model with the best parameters
        models['xgboost'] = XGBClassifier(
            **best_params,
            tree_method='hist',
            n_jobs=-1,
            random_state=42
        )
    else:
        logger.info("Hyperparameter tuning is disabled. Using default parameters.")

    # Evaluate all models
    results = {}
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        results[model_name] = evaluate_classifier(
            model, model_name, 
            X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train, y_val, y_test
        )

    # Perform SHAP analysis for each model
    for model_name, model in models.items():
        if model_name in ['xgboost', 'lightgbm']:  # SHAP supports XGBoost and LightGBM
            shap_summary_1, shap_summary_0 = shap_analysis(model, model_name, X_train_scaled, X_val_scaled, feature_cols)

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
    logger.info(f"Best performing model: {best_model_name}")

    # Train final model on full dataset with best parameters
    X_scaled = preprocessor.fit_transform(X)
    final_model = train_final_model(models[best_model_name], X_scaled, y)

    # Save best model
    model_name = f'best_model_{best_model_name}_{current_week_identifier}.joblib'
    save_model(model_path=model_output_path, model_name=model_name, model=final_model)

    # Save model features to GCS
    features_df = model_features_df[['epsilon_id', 'epcl_profileid'] + feature_cols + ['purchase_flag']]
    batch_upload_to_gcs(df=features_df, 
                        output_path=model_features_path,
                        batch_size=400000,
                        file_name=f'model_train_features_{week_identifier}',
                        fs=fs)

    # Evaluate the model for each segment
    logger.info("\nEvaluating model performance by segments...")

    segments = {
        'Segment 1 (days_since_last_purchase <= 365)': segment_1,
        'Segment 2 (730 >= days_since_last_purchase > 365)': segment_2,
        'Segment 3 (days_since_last_purchase > 0)': segment_3
    }

    segment_results = {}
    logger.info(f"Segment lengths: {len(segment_1)}, {len(segment_2)}, {len(segment_3)}")

    for segment_name, segment_df in segments.items():
        logger.info(f"\nEvaluating {segment_name}...")

        # Filter test data for the current segment
        segment_customers = segment_df['epsilon_id']
        logger.info(segment_customers.head(5))
        segment_test_indices = y_test.index[y_test.index.isin(segment_customers.index)]
        logger.info(f"Segment test indices: {segment_test_indices}")
        y_test_segment = y_test.loc[segment_test_indices]
        logger.info(f"Segment y_test: {y_test_segment.head(5)}")
        test_predictions_segment = results[best_model_name]['test_predictions'][segment_test_indices]
        logger.info(f"Segment test predictions: {test_predictions_segment.head(5)}")
        test_probabilities_segment = results[best_model_name]['test_probabilities'][segment_test_indices]
        logger.info(f"Segment test probabilities: {test_probabilities_segment.head(5)}")
        # Check if the segment has any test data
        if len(y_test_segment) == 0:
            logger.warning(f"No test data available for {segment_name}. Skipping evaluation.")
            segment_results[segment_name] = {
                'Classification Report': "No data available",
                'ROC-AUC': "No data available"
            }
            continue  # Skip the rest of the loop for this segment

        # Calculate metrics for the segment
        classification_report_segment = classification_report(y_test_segment, test_predictions_segment)
        roc_auc_segment = roc_auc_score(y_test_segment, test_probabilities_segment)

        # Log the metrics
        logger.info(f"\n{segment_name} Classification Report:\n{classification_report_segment}")
        logger.info(f"{segment_name} ROC-AUC: {roc_auc_segment:.4f}")

        # Store the results
        segment_results[segment_name] = {
            'Classification Report': classification_report_segment,
            'ROC-AUC': roc_auc_segment
        }

    # Log a summary of the results
    logger.info("\nSegment-wise Model Performance Summary:")
    for segment_name, metrics in segment_results.items():
        logger.info(f"\n{segment_name}:")
        logger.info(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")
        logger.info(f"Classification Report:\n{metrics['Classification Report']}")

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
        'top_20_shap_features_class_1': json.dumps(shap_summary_1.head(20).to_dict(orient='records')),
        'top_20_shap_features_class_0': json.dumps(shap_summary_0.head(20).to_dict(orient='records')),
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