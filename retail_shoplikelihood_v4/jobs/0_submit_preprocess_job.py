import os
from pathlib import Path
from config import settings, params
import json

from mlplatform.config.job_config import VertexAICustomDataProcJob
from mlplatform.config.gcp_config import GOOGLE_APPLICATION_CREDENTIALS
from mlplatform.utils.parser import parse_config
from mlplatform.core.vertexai.custom_job import create_custom_job, run_custom_job


class SubmitPreProcessJob:
    def __init__(self):
        pass

    
    def process(self, week_flag:str):        
        # =========================================================
        # load train configurations
        # =========================================================
    
        WEEK_FLAG = week_flag
        PROJECT_DIR = settings.PROJECT_DIR
        CURRENT_WEEK_NUMBER = params.WEEK_NUMBER
        PROJECT_PREFIX = settings.HIVE_TABLE_PREFIX

        if WEEK_FLAG == "last_four_weeks":
            WEEK_NUMBER = params.LAST_FOUR_WEEK_NUMBER
        elif WEEK_FLAG == "last_two_weeks":
            WEEK_NUMBER = params.LAST_TWO_WEEK_NUMBER

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS

        DATA_PROC_CONFIG_PATH = Path(PROJECT_DIR) / "mlmodels/ctr_shoplikelihood_v4/config/data_config.yaml"
        JOB_CONFIG: VertexAICustomDataProcJob = parse_config(DATA_PROC_CONFIG_PATH, VertexAICustomDataProcJob)

        print("project directory:", PROJECT_DIR)
        print("week number:", WEEK_NUMBER)
        print(JOB_CONFIG)
        # =========================================================
        # setting arguments, environment variables to train script
        # =========================================================
        

        lightfm_model_info = {
            "non_offer_view_first_segment_model.pickle": f"hybrid_recom_lightfm_bucket/ctr/model_registry/non_offer_view_first_segment/prod/{WEEK_NUMBER}/0/output/model/",
            "non_offer_view_second_segment_model.pickle": f"hybrid_recom_lightfm_bucket/ctr/model_registry/non_offer_view_second_segment/prod/{WEEK_NUMBER}/0/output/model/",
            "offer_view_model.pickle": f"hybrid_recom_lightfm_bucket/ctr/model_registry/offer_view/prod/{WEEK_NUMBER}/0/output/model/"
        }

        id_mapping_tables = {
            "non_offer_view_first_segment" : f"dp-lore.hybrid_recom_lightfm_ctr.non_offer_view_first_segment_recom_id_map_{WEEK_NUMBER}",
            "non_offer_view_second_segment": f"dp-lore.hybrid_recom_lightfm_ctr.non_offer_view_second_segment_recom_id_map_{WEEK_NUMBER}",
            "offer_view": f"dp-lore.hybrid_recom_lightfm_ctr.offer_view_recom_id_map_{WEEK_NUMBER}"
        }

        gcs_embeddings_output_path = f"gs://shoplikelihood_v4/ctr/{PROJECT_PREFIX}/input_data/{CURRENT_WEEK_NUMBER}/shoplikelihood_v4_customer_embeddings_ctr_{WEEK_NUMBER}"

        ENV_VARS = {
            "lightfm_model_info": json.dumps(lightfm_model_info),
            "id_mapping_tables": json.dumps(id_mapping_tables),
            "feature_segment": f"{WEEK_FLAG}_embeddings_ctr_",
            "bq_project": JOB_CONFIG.bq_project,
            "gcs_project": JOB_CONFIG.bq_project,
            "project": JOB_CONFIG.project,
            "week_identifier": WEEK_NUMBER,
            "last_week_identifier": WEEK_NUMBER,
            "gcs_embeddings_output_path": gcs_embeddings_output_path
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

submit_preprocess_job = SubmitPreProcessJob()
