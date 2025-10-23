import os
from pathlib import Path
from config import settings, params

from mlplatform.config.job_config import VertexAICustomDataProcJob
from mlplatform.config.gcp_config import GOOGLE_APPLICATION_CREDENTIALS
from mlplatform.utils.parser import parse_config
from mlplatform.core.vertexai.custom_job import create_custom_job, run_custom_job

class SubmitTrainJob:
    def __init__(self):
        pass

    
    def process(self):        
        # =========================================================
        # load train configurations
        # =========================================================

        PROJECT_DIR = settings.PROJECT_DIR
        PROJECT_PREFIX = settings.HIVE_TABLE_PREFIX

        CURRENT_WEEK_NUMBER = params.WEEK_NUMBER
        WEEK_NUMBER = params.LAST_FOUR_WEEK_NUMBER
        TARGET_WEEK_NUMBER = params.LAST_TWO_WEEK_NUMBER

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

        DATA_PROC_CONFIG_PATH = Path(PROJECT_DIR) / "mlmodels/ctr_shoplikelihood_v4/config/train.yaml"
        JOB_CONFIG: VertexAICustomDataProcJob = parse_config(DATA_PROC_CONFIG_PATH, VertexAICustomDataProcJob)

        print("project directory:", PROJECT_DIR)
        print("week number:", WEEK_NUMBER)
        print(JOB_CONFIG)
        # =========================================================
        # setting arguments, environment variables to train script
        # =========================================================
        
        predictions_output_table = f"shoplikelihood_v4.{PROJECT_PREFIX}shoplikelihood_test_predictions_{CURRENT_WEEK_NUMBER}"
        model_performance_output_table = f"shoplikelihood_v4.{PROJECT_PREFIX}shoplikelihood_test_model_performance"
        
        customer_purchase_labels_path = f"gs://shoplikelihood_v4/ctr/{PROJECT_PREFIX}/input_data/{CURRENT_WEEK_NUMBER}/shoplikelihood_v4_customer_purchase_labels_ctr_{TARGET_WEEK_NUMBER}"
        model_output_path = f"gs://shoplikelihood_v4/ctr/{PROJECT_PREFIX}/model_registry/{CURRENT_WEEK_NUMBER}/0/output/model/"
        model_features_path = f"gs://shoplikelihood_v4/ctr/{PROJECT_PREFIX}/feature_store/{CURRENT_WEEK_NUMBER}/0/output/features"
        gcs_embeddings_output_path = f"gs://shoplikelihood_v4/ctr/{PROJECT_PREFIX}/input_data/{CURRENT_WEEK_NUMBER}/shoplikelihood_v4_customer_embeddings_ctr_{WEEK_NUMBER}"
        customer_purchase_profile_path = f"gs://shoplikelihood_v4/ctr/{PROJECT_PREFIX}/input_data/{CURRENT_WEEK_NUMBER}/shoplikelihood_v4_customer_purchase_profile_ctr_{WEEK_NUMBER}"
        customer_purchase_history_path = f"gs://shoplikelihood_v4/ctr/{PROJECT_PREFIX}/input_data/{CURRENT_WEEK_NUMBER}/shoplikelihood_v4_customer_purchase_history_ctr_{WEEK_NUMBER}"
        customer_point_profile_path = f"gs://shoplikelihood_v4/ctr/{PROJECT_PREFIX}/input_data/{CURRENT_WEEK_NUMBER}/shoplikelihood_v4_customer_point_profile_ctr_{WEEK_NUMBER}" 
        npp_history_path = f"gs://shoplikelihood_v4/ctr/{PROJECT_PREFIX}/input_data/{CURRENT_WEEK_NUMBER}/shoplikelihood_v4_npp_history_ctr"
        statistic_features_path = f"gs://shoplikelihood_v4/ctr/{PROJECT_PREFIX}/input_data/{CURRENT_WEEK_NUMBER}/shoplikelihood_v4_statistic_features_ctr_{WEEK_NUMBER}"

        # Setting environment variables for the job
        ENV_VARS = {

            "feature_segment": "train_model",
            "bq_project": JOB_CONFIG.bq_project,
            "gcs_project": JOB_CONFIG.bq_project,
            "project": JOB_CONFIG.project,
            "week_identifier": WEEK_NUMBER,
            "target_week_identifier": TARGET_WEEK_NUMBER,
            "current_week_identifier": CURRENT_WEEK_NUMBER,
            "customer_purchase_labels_path": customer_purchase_labels_path,
            "customer_purchase_profile_path": customer_purchase_profile_path,
            "customer_purchase_history_path": customer_purchase_history_path,
            "customer_point_profile_path": customer_point_profile_path,
            "npp_history_path": npp_history_path,
            "statistic_features_path": statistic_features_path,
            "model_output_path": model_output_path,
            "model_features_path": model_features_path,
            "predictions_output_table": predictions_output_table,
            "gcs_embeddings_output_path": gcs_embeddings_output_path,
            "model_performance_output_table": model_performance_output_table
        }

        if JOB_CONFIG.is_enabled:

            data_preprocess_job = create_custom_job(
                display_name          = JOB_CONFIG.display_name+ENV_VARS['feature_segment'],
                script_path           = JOB_CONFIG.script_path,
                container_uri         = JOB_CONFIG.container_uri,
                requirements          = JOB_CONFIG.requirements,
                environment_variables = ENV_VARS,
                replica_count         = JOB_CONFIG.replica_count,
                machine_type          = JOB_CONFIG.machine_type,
                accelerator_type      = JOB_CONFIG.accelerator_type,
                accelerator_count     = JOB_CONFIG.accelerator_count,
                boot_disk_type        = JOB_CONFIG.boot_disk_type,
                boot_disk_size_gb     = JOB_CONFIG.boot_disk_size_gb,
                project               = JOB_CONFIG.project,
                location              = JOB_CONFIG.location,
                staging_bucket        = JOB_CONFIG.staging_bucket,
            )

            run_custom_job(
                job=data_preprocess_job,
                week_identifier=WEEK_NUMBER,
                owner=JOB_CONFIG.owner,
                job_type=JOB_CONFIG.type,
                bq_project=JOB_CONFIG.bq_project,
                log_table_path=JOB_CONFIG.job_metadata_table_path,
                timeout=JOB_CONFIG.timeout,
                enable_web_access=JOB_CONFIG.enable_web_access,
                sync=True
            )

submit_training_job = SubmitTrainJob()
