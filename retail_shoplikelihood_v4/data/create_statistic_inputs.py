from app.util.bigquery.connection import SparkBQ
from pyspark.sql.functions import col, when, lit, current_date, to_date,coalesce, md5, trim, coalesce, upper, col, current_date, date_format, date_sub
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import argparse
from create_customer_features import PrepareCustomerFeatures
from datetime import timedelta
import logging
import sys

#input
from app.io.source import loyalty_account_current, pca_dashboard_bigquery
from app.io.sink import active_customers

#output
from app.io.sink import shoplikelihood_v4_customer_base_ctr, shoplikelihood_v4_transaction_base_ctr, shoplikelihood_v4_offer_views_ctr

BANNER_list = ['CTR']
active_customers.table_prefix = 'prd_v2_'


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class CreateStatisticInputs(SparkBQ):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = self.get_args()
        self.week_num = self.args.week_identifier
        self.date_3w = None
        self.end_date = None
        self.start_date = None
        self.prepare_customer_features = PrepareCustomerFeatures()
        self.reco_reference_df = self.prepare_customer_features.reco_reference_df \
                                     .withColumn("start_dateid", date_format(col("start_date"), 'yyyyMMdd').cast(StringType())) \
                                     .withColumn("end_dateid", date_format(col("end_date"), 'yyyyMMdd').cast(StringType()))
        self.week_identifier = self.prepare_customer_features.week_identifier

    def get_args(self):
        parser = argparse.ArgumentParser(description='Create Statistic Features')
        parser.add_argument('--week_identifier', required=True, type=str, help='Week identifier in the format YYYY_3WW')
        return parser.parse_args()
    

    @property
    def customer_base_df(self):
        return active_customers.read(spark=self.spark).withColumnRenamed("customer_id", "customerid")\
                               .crossJoin(self.reco_reference_df \
                                    .select('deal_yr', 'deal_wk', 'start_date', 'end_date', 'week_identifier') \
                                    .filter(col('end_date') > date_sub(current_date(), 372)            )) \
                                    .withColumn('p_banner', lit('CTR')) \
                                    .select('customerid', 'deal_yr', 'deal_wk', 'start_date', 'end_date', 'week_identifier', 'p_banner') \
                                    .cache()
                            
    def create_offer_view_df(self, date_begin, date_end):
        active_customers_df = active_customers.read(spark=self.spark).select(col("customer_id").alias("customerid")) \
                                            .withColumn('epsilon_id', md5(upper(trim(coalesce((col('customerid').cast(StringType())))))))

        query =  f"""
                SELECT 
                    c.customer_id_hashed AS epsilon_id,  
                    b.triangle_website_view_date, 
                    b.triangle_app_view_date, 
                    b.p_banner,
                    b.email_view_date,
                    b.ctr_website_view_date,
                    b.ctr_odp_website_view,
                    b.ctr_app_view_date,
                    b.p_trans_dateid as view_date
                FROM `customer-analytics-306513.PCA_Dashboard.p0001_pca_main` b
                INNER JOIN `customer-analytics-306513.cust_analytics_prod.gc_epcl_mapping` c
                    ON b.customerid = c.epcl_id
                WHERE b.p_trans_dateid >= "{date_begin}"
                    AND b.p_trans_dateid < "{date_end}"
                    AND (
                        b.triangle_app_view_date IS NOT NULL 
                        OR b.triangle_website_view_date IS NOT NULL 
                        OR b.email_view_date IS NOT NULL
                        OR b.ctr_website_view_date IS NOT NULL
                        OR b.ctr_odp_website_view IS NOT NULL
                        OR b.ctr_app_view_date IS NOT NULL
                    )
                    AND b.p_banner IN ('CTR')
                """
        offer_view_df = self.read_from_bq(path_to_bq_table=pca_dashboard_bigquery.table_path, bq_query=query) \
                            .join(active_customers_df.select(col("customerid"), col("epsilon_id")),
                                  on="epsilon_id",
                                  how="inner") \
                            .distinct() \
                            .withColumn("view_ind",
                                when(
                                    col("triangle_website_view_date").isNotNull() |
                                    col("triangle_app_view_date").isNotNull() |
                                    col("email_view_date").isNotNull() |
                                    col("ctr_website_view_date").isNotNull() |
                                    col("ctr_odp_website_view").isNotNull() |
                                    col("ctr_app_view_date").isNotNull(), 1).otherwise(0)) \
                            .select("customerid", "view_ind", "p_banner", "view_date").distinct().cache()

        offer_view_df.show()
        return offer_view_df


    def process(self):
        #get the date range for the last 3 years
        end_date = self.reco_reference_df.filter(col("week_identifier") == self.week_identifier).first().start_date
        start_date = end_date - timedelta(days=1095)
        date_begin = start_date.strftime("%Y-%m-%d")
        date_end = end_date.strftime("%Y-%m-%d")
        dateid_begin = start_date.strftime("%Y%m%d")
        dateid_end = end_date.strftime("%Y%m%d")
        
        

        loyalty_account_df = loyalty_account_current.read(spark=self.spark)\
                                                    .withColumn('epsilon_id', md5(upper(trim(coalesce((col('customerid').cast(StringType()))))))) \
                                                    .select("customerid", "epcl_profileid", "epsilon_id", "cardnumber")
        offer_view_df = self.create_offer_view_df(date_begin, date_end)
        offer_view_ref_df = offer_view_df.join(self.reco_reference_df,
                                        (offer_view_df.view_date >= self.reco_reference_df.start_date) &
                                        (offer_view_df.view_date <= self.reco_reference_df.end_date),
                                        how="inner")
        logger.info("Saving offer_view dataframe:")
        logger.info(f"DF columns: {offer_view_ref_df.columns}")
        shoplikelihood_v4_offer_views_ctr.save(df=offer_view_ref_df)        

        transaction_detail_df = self.prepare_customer_features.load_transaction_data(dateid_begin, dateid_end) \
                                                              .join(loyalty_account_df, on=["cardnumber"], how="inner") \
                                                              .withColumn("transaction_date", to_date(col("tx_ts"))) \
                                                              .select("customerid", "p_trans_dateid", "transaction_date", "banner", "lms_transaction_id", 
                                                                    "purchase_flag", "sales", "tx_ts", "tx_epoch", "prev_tx_ts", "current_date" )\
                                                              .withColumn("purchase_flag", lit(1)) \
                                                              .withColumnRenamed("banner", "p_banner")
        
        transaction_detail_ref_df = transaction_detail_df.join(self.reco_reference_df,
                                                            (transaction_detail_df.p_trans_dateid >= self.reco_reference_df.start_dateid) &
                                                            (transaction_detail_df.p_trans_dateid <= self.reco_reference_df.end_dateid),
                                                            how="inner").cache() 
        
        logger.info("Saving transaction_detail dataframe:")
        logger.info(f"DF columns: {transaction_detail_ref_df.columns}")
        shoplikelihood_v4_transaction_base_ctr.save(df=transaction_detail_ref_df)

        # Alias the DataFrames to avoid column name conflicts
        customer_base_df = self.customer_base_df
        logger.info("Saving customer_base dataframe:")
        shoplikelihood_v4_customer_base_ctr.save(df=customer_base_df)


if __name__ == '__main__':
    create_statistic_features = CreateStatisticInputs()
    create_statistic_features.process()


