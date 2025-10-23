import gcsfs
import json
import os
import numpy as np
import pandas as pd
import pickle
from google.cloud import bigquery
import logging
import sys

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def load_model_emb(model_path, model_name):
    """Load LightFM model and extract user embeddings."""
    logger.info(f"Loading model: {model_name} from {model_path}")
    fs = gcsfs.GCSFileSystem()
    full_path = model_path + model_name
    with fs.open(full_path, 'rb') as f:
        model = pickle.load(f)

    user_bias, user_emb = model.get_user_representations()  # ndarray (n_users, k)
    n_instances, k = user_emb.shape

    emb_df = pd.DataFrame(user_emb, columns=[f"latent_{i}" for i in range(k)])
    emb_df["lightfm_id"] = np.arange(n_instances)
    logger.info(emb_df.head())
    return emb_df, n_instances, k


def batch_upload_to_gcs(df, output_path, batch_size, fs):
    """Upload DataFrame to GCS in parquet format with batching."""
    logger.info(f"Uploading {len(df)} rows to {output_path} in batches of {batch_size}...")
    
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        chunk = df.iloc[start:end]
        
        # Create unique filename for this chunk
        chunk_filename = f"embeddings_part_{start}_{end}.parquet"
        full_path = os.path.join(output_path, chunk_filename)
        
        logger.info(f"Writing chunk {start} to {end} to {full_path}")
        with fs.open(full_path, 'wb') as f:
            chunk.to_parquet(f, index=False)
    
    logger.info("All batches uploaded successfully to GCS.")


if __name__ == "__main__":

    # Load environment variables
    lightfm_model_info = json.loads(os.environ.get('lightfm_model_info'))
    id_mapping_tables = json.loads(os.environ.get('id_mapping_tables'))
    bq_project = os.environ.get('bq_project')
    gcs_project = os.environ.get('gcs_project')
    gcs_output_path = os.environ.get('gcs_embeddings_output_path')
    week_identifier = os.environ.get('week_identifier')

    # Initialize clients
    bq_client = bigquery.Client(project=bq_project)
    fs = gcsfs.GCSFileSystem(project=gcs_project)
    batch_size = 400000

   
    customer_embeddings_list = []

    model_names = lightfm_model_info.keys()
    for lightfm_model_name in model_names:
        lighfm_model_segment = lightfm_model_name.replace("_model.pickle", "")

         # Load model embeddings
        lightfm_model_path = lightfm_model_info[lightfm_model_name]
        emb_df, n_instances, k = load_model_emb(model_path=lightfm_model_path, model_name=lightfm_model_name)
        


        # Read the id mapping table from BigQuery
        id_mapping_table = id_mapping_tables[lighfm_model_segment]
        query = f"SELECT user_id, epsilon_id, lightfm_id FROM `{id_mapping_table}`"
        map_df = bq_client.query(query).to_dataframe()
        n_users = len(map_df)
        logger.info(map_df.head())
        n_user_features = n_instances - n_users

        #print out the number of users and user features
        logger.info(f"✔ {lighfm_model_segment}  model loaded   ‑ {n_instances} instances, {k}‑dim embeddings, {n_users} users, {n_user_features} user_features ")

        #Merge and append results to list
        logger.info(f"Embeddings cols: {emb_df.columns.tolist()}")
        logger.info(f"Map cols: {map_df.columns.tolist()}")

        result_emb_df = emb_df.merge(map_df, on="lightfm_id", how="inner")
        logger.info(f"Result Embeddings cols: {result_emb_df.columns}")
        customer_embeddings_list.append(result_emb_df)


    # Concatenate all model embeddings into a single DataFrame
    final_emb_df = pd.concat(customer_embeddings_list, ignore_index=True)
    logger.info(f"Final Embeddings cols: {final_emb_df.columns}")

    # Upload the final embeddings DataFrame to GCS
    batch_upload_to_gcs(
        df=final_emb_df,
        output_path=gcs_output_path,
        batch_size=batch_size,
        fs=fs
    )
    logger.info("Embeddings saved to GCS successfully.")