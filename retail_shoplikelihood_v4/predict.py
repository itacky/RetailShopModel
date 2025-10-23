import logging
import sys
import os
from google.cloud import bigquery
from pyarrow import parquet
import gcsfs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


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


def batch_upload_to_bigquery(batch_size, df, table_name, source_format, bq_client):
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



if __name__ == "__main__":
    logger.info("Loading environment variables...")
    gcs_embeddings_output_path = os.environ.get('gcs_embeddings_output_path')
    predictions_output_table = os.environ.get('predictions_output_table')
    customer_purchase_profile_path = os.environ.get('customer_purchase_profile_path')
    customer_purchase_history_path = os.environ.get('customer_purchase_history_path')
    customer_point_profile_path = os.environ.get('customer_point_profile_path')
    npp_history_path = os.environ.get('npp_history_path')
    week_identifier = os.environ.get('week_identifier')
    target_week_identifier = os.environ.get('target_week_identifier')
    statistic_features_path = os.environ.get('statistic_features_path')
    model_output_path = os.environ.get('model_output_path')
    tlc_segment = 'Onboarding'



    logger.info("Starting bigquery client and GCS file system...")
    bq_project = os.environ.get('bq_project')
    bq_client = bigquery.Client(project=bq_project)
    fs               = gcsfs.GCSFileSystem()

    #==============================================
    #Load Model from model output 
    #==============================================
    model_name = f'best_model_xgboost_{target_week_identifier}.joblib'
    logger.info(f"Loading model from: {model_output_path}")
    model_path = os.path.join(model_output_path, model_name)
    fs = gcsfs.GCSFileSystem()
    with fs.open(model_path, 'rb') as f:
        model = joblib.load(f)
    logger.info(f"Model loaded successfully: {model_name}")
    #==============================================
    # Load customer features
    #==============================================
    reco_reference_pd = load_reco_reference_df(bq_client=bq_client)
    

    #triangle_lifecycle_segment data
    tlc_end_date = reco_reference_pd[reco_reference_pd["week_identifier"] == week_identifier].end_date.iloc[0]
    tlc_segments_df = load_triangle_lifecycle_segment_df(bq_client=bq_client, tlc_date=tlc_end_date, segment=tlc_segment)
    logger.info(f"Loaded {len(tlc_segments_df)} triangle lifecycle segment records for segment '{tlc_segment}'. Date: {tlc_end_date}")

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
    npp_history_pd_df = npp_history_pd_df[npp_history_pd_df["week_identifier"] <=week_identifier]
    sorted_unique_week_identifiers = npp_history_pd_df['week_identifier'].drop_duplicates().sort_values()
    smallest_week_identifier = sorted_unique_week_identifiers.iloc[0]
    second_smallest_week_identifier = sorted_unique_week_identifiers.iloc[1]
    logger.info(f"Excluding weeks: {smallest_week_identifier}, {second_smallest_week_identifier} from NPP features.")
    npp_history_pd_df = npp_history_pd_df[
        (npp_history_pd_df["week_identifier"] != smallest_week_identifier) & 
        (npp_history_pd_df["week_identifier"] != second_smallest_week_identifier)
    ]
    npp_history_pivoted_df = npp_history_pd_df.set_index('week_identifier')['npp_flag'].T.to_frame().T
    npp_history_pivoted_df.reset_index(drop=True, inplace=True)
    npp_history_pivoted_df = rename_pivot_columns(npp_history_pivoted_df, reco_reference_pd, week_identifier, "_weeks_back_NPP")
    npp_history_pivoted_df = npp_history_pivoted_df.add_suffix('_npp_flag')




    # Combine Model Features DataFrame
    logger.info(embedding_df.columns.tolist())
    logger.info(customer_profile_summary_df.columns.tolist())
    model_features_df = embedding_df.merge(customer_profile_summary_df, on='epsilon_id', how='inner') \
                                    .merge(npp_history_pivoted_df, how='cross') \
                                    .drop_duplicates(subset='epsilon_id')
    
    logger.info(f"Model features shape: {model_features_df.shape}")
    logger.info(model_features_df.head())
    
    # =============================================
    # Prepare data for training
    # =============================================
    feature_cols = model_features_df.columns.drop(["epsilon_id", "epcl_profileid", "lightfm_id", "user_id", "enroll_date", "register_date", "first_purchase_date", "last_purchase_date"]).tolist()
    logger.info(f"Feature columns used for training: {feature_cols}")

    # Prepare X (features) and y (target)
    X = model_features_df[feature_cols]

     # Detect binary and numeric columns
    column_vals = ["weeks_back", "deal_season"]
    binary_cols = []
    for column in X.columns:
        if any (val in column for val in column_vals):
            binary_cols.append(column)

    numeric_cols = [col for col in X.columns if col not in binary_cols]
    logger.info(f"Numeric columns: {numeric_cols}")
    logger.info(f"Binary columns: {binary_cols}")



    # Fit scaler only on training data
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('bin', 'passthrough', binary_cols)
    ])


    # Scale the training data
    X_scaled = preprocessor.fit_transform(X)


   
    # =============================================
    # Make predictions
    # =============================================
    model_features_df['predicted_purchase_prob'] = model.predict_proba(X_scaled)[:, 1]

    batch_size = 400000  
    # Save predictions with customer IDs
    predictions_df = model_features_df[['epsilon_id', 'epcl_profileid', 'predicted_purchase_prob']]
    batch_upload_to_bigquery(
        batch_size=batch_size, 
        df=predictions_df, 
        table_name=predictions_output_table, 
        source_format=bigquery.SourceFormat.PARQUET, 
        bq_client=bq_client
    )
    
    logger.info(f"Predictions saved to table: {predictions_output_table}")
    logger.info("Model comparison and evaluation completed successfully.")

