from airflow_helper.task import resolve_spark_task_with_whl_and_extra_packages, resolve_job_path
import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator

from airflow_helper.task import import_num_modules

from airflow.operators.bash_operator import BashOperator
from mlplatform.feature_store.batch.hdfs_to_gcs import get_ingest_table_cmd
from mlplatform.feature_store.batch.hdfs_create_directory_on_gcs import create_directory_cmd
from app.io.sink import shoplikelihood_v4_customer_purchase_labels_ctr, shoplikelihood_v4_customer_purchase_history_ctr, shoplikelihood_v4_customer_purchase_profile_ctr, shoplikelihood_v4_customer_point_profile_ctr, shoplikelihood_v4_statistic_features_ctr, shoplikelihood_v4_npp_history_ctr
from mlmodels.ctr_shoplikelihood_v4.data.create_adhoc_offer_table import CreateAdHocOfferInfoFromCSV
from config import params

week_identifier = params.WEEK_NUMBER
last_two_week_identifier = params.LAST_TWO_WEEK_NUMBER
last_four_week_identifier = params.LAST_FOUR_WEEK_NUMBER
last_week_identifier = params.LAST_WEEK_NUMBER
gcs_project_id = 'dp-lore'
npp_history_flag = 'Y'

shoplikelihood_v4_statistic_features_ctr_name = shoplikelihood_v4_statistic_features_ctr(week_identifier=week_identifier).resolve_table_name

last_two_week_shoplikelihood_v4_customer_purchase_labels_ctr_table_name = shoplikelihood_v4_customer_purchase_labels_ctr(week_identifier=last_two_week_identifier).resolve_table_name
last_four_week_shoplikelihood_v4_customer_purchase_history_ctr_name = shoplikelihood_v4_customer_purchase_history_ctr(week_identifier=last_four_week_identifier).resolve_table_name
last_four_week_shoplikelihood_v4_customer_purchase_profile_ctr_name = shoplikelihood_v4_customer_purchase_profile_ctr(week_identifier=last_four_week_identifier).resolve_table_name
last_four_week_shoplikelihood_v4_customer_point_profile_ctr_name = shoplikelihood_v4_customer_point_profile_ctr(week_identifier=last_four_week_identifier).resolve_table_name
shoplikelihood_v4_npp_history_ctr_name = shoplikelihood_v4_npp_history_ctr.resolve_table_name

last_two_week_shoplikelihood_v4_customer_purchase_labels_ctr_table_name = shoplikelihood_v4_customer_purchase_labels_ctr(week_identifier=last_two_week_identifier).resolve_table_name
last_two_week_shoplikelihood_v4_customer_purchase_history_ctr_name = shoplikelihood_v4_customer_purchase_history_ctr(week_identifier=last_two_week_identifier).resolve_table_name
last_two_week_shoplikelihood_v4_customer_purchase_profile_ctr_name = shoplikelihood_v4_customer_purchase_profile_ctr(week_identifier=last_two_week_identifier).resolve_table_name
last_two_week_shoplikelihood_v4_customer_point_profile_ctr_name = shoplikelihood_v4_customer_point_profile_ctr(week_identifier=last_two_week_identifier).resolve_table_name
 

#GCP Tasks
_submit_preprocess_job = import_num_modules('mlmodels/ctr_shoplikelihood_v4/jobs/0_submit_preprocess_job.py', 'preprocess_module').SubmitPreProcessJob()
_submit_training_job = import_num_modules('mlmodels/ctr_shoplikelihood_v4/jobs/1_submit_train_job.py', 'preprocess_module').SubmitTrainJob()
_submit_predict_job = import_num_modules('mlmodels/ctr_shoplikelihood_v4/jobs/2_submit_predict_job.py', 'preprocess_module').SubmitPredictJob()
create_ad_hoc_offer_info = CreateAdHocOfferInfoFromCSV()

desire_exec_week_day = 7  # Monday
deployment_week_day = datetime.today().date().isoweekday()
start_date = datetime.now() - timedelta(days=8 - abs(desire_exec_week_day - deployment_week_day))
default_args = {
    'owner': 'loyalty_reco',
    'start_date': start_date,

}

with DAG(
        dag_id='shoplikelihood_v4_CTR',
        description='Data Preprocessing for CTR Customer Segmentation model',
        default_args=default_args,
        schedule_interval=None,
        catchup=False,
        max_active_runs = 1
) as dag:

    
    _generate_last_two_weeks_lightfm_embeddings =  PythonOperator(
        task_id='generate_last_two_weeks_lightfm_embeddings',
        python_callable=_submit_preprocess_job.process,
        op_args=["last_two_weeks"], #week_flag="last_two_weeks"
        execution_timeout=None)
    
    _generate_last_four_week_lightfm_embeddings =  PythonOperator(
        task_id='generate_last_four_week_lightfm_embeddings',
        python_callable=_submit_preprocess_job.process,
        op_args=["last_four_weeks"], #week_flag="last_four_week"
        execution_timeout=None)

    # _create_customer_labels=  PythonOperator(
    #     task_id='create_customer_labels',
    #     python_callable=_create_customer_labels.process,
    #     execution_timeout=None)

    _create_ad_hoc_offer_info = PythonOperator(
        task_id='create_ad_hoc_offer_info',
        python_callable=create_ad_hoc_offer_info.process,
        execution_timeout=None
    )

    _save_adhoc_offer_info = resolve_spark_task_with_whl_and_extra_packages(
        task_id='save_adhoc_offer_info',
        job_path=resolve_job_path(folder='mlmodels/ctr_shoplikelihood_v4/data', 
                                  script='save_adhoc_offer_table.py'),
        execution_timeout=None
    )

    _create_statistic_inputs = resolve_spark_task_with_whl_and_extra_packages(
        task_id='create_statistic_inputs',
        job_path=resolve_job_path(folder='mlmodels/ctr_shoplikelihood_v4/data', 
                                  script='create_statistic_inputs.py'),
        job_input_params=f"--week_identifier={week_identifier}",
        execution_timeout=None
    )
    
    
    _prepare_statistic_features = resolve_spark_task_with_whl_and_extra_packages(
        task_id='prepare_statistic_features',
        job_path=resolve_job_path(folder='mlmodels/ctr_shoplikelihood_v4/data', 
                                  script='create_statistic_features.py'),
        job_input_params=f"--week_identifier={week_identifier}",
        execution_timeout=None
    )


    _prepare_customer_purchase_labels_input = resolve_spark_task_with_whl_and_extra_packages(
        task_id='prepare_customer_purchase_labels_input',
        job_path=resolve_job_path(folder='mlmodels/ctr_shoplikelihood_v4/data', 
                                  script='create_customer_labels.py'),
        execution_timeout=None
    )

    _prepare_last_two_week_customer_features = resolve_spark_task_with_whl_and_extra_packages(
        task_id='prepare_last_two_week_customer_features',
        job_path=resolve_job_path(folder='mlmodels/ctr_shoplikelihood_v4/data', 
                                  script='create_customer_features.py'),
        job_input_params=f"--week_identifier={last_two_week_identifier} --npp_history_flag={npp_history_flag}",
        execution_timeout=None
    )

    _prepare_last_four_week_customer_features = resolve_spark_task_with_whl_and_extra_packages(
        task_id='prepare_last_four_week_customer_features',
        job_path=resolve_job_path(folder='mlmodels/ctr_shoplikelihood_v4/data', 
                                  script='create_customer_features.py'),
        job_input_params=f"--week_identifier={last_four_week_identifier}",
        execution_timeout=None
    )

    _create_input_data_folder = BashOperator(
        task_id = 'create_input_data_folder_on_gcs',
        bash_command = create_directory_cmd(
            project_id = gcs_project_id,
            bucket = shoplikelihood_v4_customer_purchase_profile_ctr.gcs_bucket,
            new_folder = week_identifier
        ),
        retries = 3,
    )


    _ingest_shoplikelihood_v4_customer_purchase_labels_ctr_to_gcs = BashOperator(
        task_id = 'ingest_shoplikelihood_v4_customer_purchase_labels_ctr_to_gcs',
        bash_command = get_ingest_table_cmd(
            last_two_week_shoplikelihood_v4_customer_purchase_labels_ctr_table_name,
            os.path.join(shoplikelihood_v4_customer_purchase_labels_ctr.gcs_bucket +"/"+ week_identifier, last_two_week_shoplikelihood_v4_customer_purchase_labels_ctr_table_name)
        ),
        retries = 3,
    )

    _ingest_last_two_week_customer_purchase_history_ctr_to_gcs = BashOperator(
        task_id = 'ingest_last_two_week_customer_purchase_history_ctr_to_gcs',
        bash_command = get_ingest_table_cmd(
            last_two_week_shoplikelihood_v4_customer_purchase_history_ctr_name,
            os.path.join(shoplikelihood_v4_customer_purchase_history_ctr.gcs_bucket +"/"+ week_identifier, last_two_week_shoplikelihood_v4_customer_purchase_history_ctr_name)
        ),
        retries = 3,
    )


    _ingest_last_four_week_customer_purchase_history_ctr_to_gcs = BashOperator(
        task_id = 'ingest_last_four_week_customer_purchase_history_ctr_to_gcs',
        bash_command = get_ingest_table_cmd(
            last_four_week_shoplikelihood_v4_customer_purchase_history_ctr_name,
            os.path.join(shoplikelihood_v4_customer_purchase_history_ctr.gcs_bucket +"/"+ week_identifier, last_four_week_shoplikelihood_v4_customer_purchase_history_ctr_name)
        ),
        retries = 3,
    )

    
    _ingest_lask_two_week_customer_purchase_profile_ctr_to_gcs = BashOperator(
        task_id = 'ingest_last_two_week_customer_purchase_profile_ctr_to_gcs',  
        bash_command = get_ingest_table_cmd(
            last_two_week_shoplikelihood_v4_customer_purchase_profile_ctr_name,
            os.path.join(shoplikelihood_v4_customer_purchase_profile_ctr.gcs_bucket +"/"+ week_identifier, last_two_week_shoplikelihood_v4_customer_purchase_profile_ctr_name)
        ),
        retries = 3,
    )


    _ingest_last_four_week_customer_purchase_profile_ctr_to_gcs = BashOperator(
        task_id = 'ingest_last_four_week_customer_purchase_profile_ctr_to_gcs',  
        bash_command = get_ingest_table_cmd(
            last_four_week_shoplikelihood_v4_customer_purchase_profile_ctr_name,
            os.path.join(shoplikelihood_v4_customer_purchase_profile_ctr.gcs_bucket +"/"+ week_identifier, last_four_week_shoplikelihood_v4_customer_purchase_profile_ctr_name)
        ),
        retries = 3,
    )

    _ingest_last_two_week_customer_point_profile_ctr_to_gcs = BashOperator(
        task_id = 'ingest_last_two_week_customer_point_profile_ctr_to_gcs',
        bash_command = get_ingest_table_cmd(
            last_two_week_shoplikelihood_v4_customer_point_profile_ctr_name,
            os.path.join(shoplikelihood_v4_customer_point_profile_ctr.gcs_bucket +"/"+ week_identifier, last_two_week_shoplikelihood_v4_customer_point_profile_ctr_name)
        ),
        retries = 3,
    )
    

    _ingest_last_four_week_customer_point_profile_ctr_to_gcs = BashOperator(
        task_id = 'ingest_last_four_week_customer_point_profile_ctr_to_gcs',
        bash_command = get_ingest_table_cmd(
            last_four_week_shoplikelihood_v4_customer_point_profile_ctr_name,
            os.path.join(shoplikelihood_v4_customer_point_profile_ctr.gcs_bucket +"/"+ week_identifier, last_four_week_shoplikelihood_v4_customer_point_profile_ctr_name)
        ),
        retries = 3,
    )


    _ingest_npp_history_ctr_to_gcs = BashOperator(
        task_id = 'ingest_npp_history_ctr_to_gcs',
        bash_command = get_ingest_table_cmd(
            shoplikelihood_v4_npp_history_ctr_name,
            os.path.join(shoplikelihood_v4_npp_history_ctr.gcs_bucket +"/"+ week_identifier, shoplikelihood_v4_npp_history_ctr_name)
        ),  
        retries = 3,
    )


    # _ingest_statistic_features_ctr_to_gcs = BashOperator(
    #     task_id = 'ingest_statistic_features_ctr_to_gcs',
    #     bash_command = get_ingest_table_cmd(
    #         shoplikelihood_v4_statistic_features_ctr_name,
    #         os.path.join(shoplikelihood_v4_statistic_features_ctr.gcs_bucket +"/"+ week_identifier, shoplikelihood_v4_statistic_features_ctr_name)
    #     ),
    #     retries = 3,
    # )


    _train_model =  PythonOperator(
        task_id='train_shoplikelihood_v4_model',
        python_callable=_submit_training_job.process,
        execution_timeout=None)
    

    _make_predictions =  PythonOperator(
        task_id='predict_shoplikelihood_v4_model',
        python_callable=_submit_predict_job.process,
        execution_timeout=None)
    

    _create_shoplikelihood_labels = resolve_spark_task_with_whl_and_extra_packages(
        task_id='create_shoplikelihood_labels',
        job_path=resolve_job_path(folder='mlmodels/ctr_shoplikelihood_v4/data', 
                                  script='create_likelihood_labels.py'),
        execution_timeout=None
    )
    
    _create_final_output = resolve_spark_task_with_whl_and_extra_packages(
        task_id='create_final_output',
        job_path=resolve_job_path(folder='mlmodels/ctr_shoplikelihood_v4/data',
                                    script='merge_final_labels.py'),
        execution_timeout=None
    )
    
    # _start_pipeline = DummyOperator(
    #     task_id='start_pipeline',
    #     execution_timeout=None)
    
    _end_pipeline = DummyOperator(
        task_id='end_pipeline',
        execution_timeout=None)
    
    

_prepare_customer_features =        [_prepare_last_two_week_customer_features, _prepare_last_four_week_customer_features]
_generate_lightfm_embeddings =      [_generate_last_four_week_lightfm_embeddings, _generate_last_two_weeks_lightfm_embeddings]
_last_two_week_ingestion_tasks =    [_ingest_last_two_week_customer_purchase_history_ctr_to_gcs,
                                     _ingest_lask_two_week_customer_purchase_profile_ctr_to_gcs,
                                     _ingest_last_two_week_customer_point_profile_ctr_to_gcs]
_last_four_week_ingestion_tasks =        [_ingest_last_four_week_customer_purchase_history_ctr_to_gcs,
                                     _ingest_last_four_week_customer_purchase_profile_ctr_to_gcs,
                                     _ingest_last_four_week_customer_point_profile_ctr_to_gcs,
                                     _ingest_npp_history_ctr_to_gcs]

#Model Pipeline Tasks
# initial_tasks = [create_ad_hoc_offer_info, _create_input_data_folder, _prepare_statistic_features, _prepare_customer_purchase_labels_input]
# _start_pipeline >> initial_tasks


_prepare_last_four_week_customer_features >> _last_four_week_ingestion_tasks >> _train_model
_generate_last_two_weeks_lightfm_embeddings >> _train_model >> _make_predictions

_create_ad_hoc_offer_info >> _save_adhoc_offer_info >> _prepare_customer_features
_create_input_data_folder >> _generate_lightfm_embeddings 
_prepare_customer_purchase_labels_input >> _ingest_shoplikelihood_v4_customer_purchase_labels_ctr_to_gcs >> _train_model
[_prepare_last_two_week_customer_features, _prepare_last_four_week_customer_features] >> _create_statistic_inputs >> _prepare_statistic_features >> _create_shoplikelihood_labels

_generate_last_four_week_lightfm_embeddings >> _make_predictions
_prepare_last_two_week_customer_features >> _last_two_week_ingestion_tasks >> _make_predictions  >> _create_shoplikelihood_labels >> _create_final_output >> _end_pipeline

