from app.util.bigquery.connection import SparkBQ
from create_customer_features import PrepareCustomerFeatures
from datetime import timedelta
from pyspark.sql.functions import col, when, lit, coalesce, md5, trim, coalesce, upper, col, date_format, countDistinct
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
import argparse
import logging
import sys

#input
from app.io.sink import active_customers, shoplikelihood_v4_customer_base_ctr, shoplikelihood_v4_transaction_base_ctr, shoplikelihood_v4_offer_views_ctr

#output
from app.io.sink import shoplikelihood_v4_statistic_features_ctr

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

class CreateStatisticFeatures(SparkBQ):
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
    

    def process(self):
        #get the date range for the last 3 years
        end_date = self.reco_reference_df.filter(col("week_identifier") == self.week_identifier).first().start_date
        todays_start_date = end_date
        cutoff_date = end_date - timedelta(days=365)
        

        customer_base_df = shoplikelihood_v4_customer_base_ctr.read(spark=self.spark).alias("cb")
        transaction_detail_ref_df = shoplikelihood_v4_transaction_base_ctr.read(spark=self.spark).alias("td")
        offer_view_ref_df = shoplikelihood_v4_offer_views_ctr.read(spark=self.spark).alias("ov") \
                                                             .dropDuplicates(['customerid', 'week_identifier'])


        # Perform the joins with explicit column references
        agg_transaction_df = customer_base_df.join(transaction_detail_ref_df, on=['customerid', 'deal_yr', 'deal_wk', 'p_banner'], how='left') \
                                             .join(offer_view_ref_df, on=['customerid', 'deal_yr', 'deal_wk', 'p_banner'], how='left') \
                                             .select(
                                                 col("cb.customerid").alias("customerid"),
                                                 col("cb.p_banner").alias("p_banner"),
                                                 col("cb.deal_yr").alias("deal_yr"),
                                                 col("cb.deal_wk").alias("deal_wk"),
                                                 col("cb.start_date").alias("start_date"),
                                                 col("cb.end_date").alias("end_date"),
                                                 col("td.transaction_date").alias("transaction_date"),
                                                 col("td.purchase_flag").alias("shop_ind"),
                                                 col("ov.view_ind").alias("view_ind")
                                             ) \
                                             .groupBy("customerid", "p_banner", "deal_yr", "deal_wk", "start_date", "end_date") \
                                             .agg(F.max('transaction_date').alias('last_shop_date'),
                                                  F.max(coalesce(col('shop_ind'), lit(0))).alias('weekly_shop_ind'),
                                                  F.max(coalesce(col('view_ind'), lit(0))).alias('weekly_view_ind'),
                                                  countDistinct('transaction_date').alias('weekly_trips'),
                                                  F.max(when((coalesce(col('shop_ind'), lit(0)) == 1 ) & (coalesce('view_ind', lit(0)) == 1), 1).otherwise(0)).alias('weekly_shop_view_ind')
                                             ).filter(col('start_date').between(cutoff_date, end_date)).cache()
        agg_transaction_df.show()
        

        customer_df = customer_base_df.filter(col('start_date') == todays_start_date)\
                                      .select('customerid', 'p_banner', 'start_date', 'end_date', 'deal_yr', 'deal_wk')
        customer_df.show()
        print(customer_df.count())

        aided_stats_df = agg_transaction_df.groupBy('customerid', 'p_banner') \
                                           .agg(F.sum(col('weekly_view_ind')).alias('weeks_views'),
                                          F.sum(col('weekly_shop_view_ind')).alias('shop_view_weeks'),
                                         (F.sum(col('weekly_view_ind')) / 52).alias('lambda_views')).cache()
        aided_stats_df.show()
        print(aided_stats_df.count())
        agg_transaction_df.unpersist()

        earliest_transaction_df = transaction_detail_ref_df.groupBy('customerid', 'p_banner') \
                                                           .agg(F.min('transaction_date').alias("earliest_transaction_date"))
        earliest_transaction_df.show()
        print(earliest_transaction_df.count())
        transaction_detail_ref_df.unpersist()
        
        result_df = customer_df.join(aided_stats_df, on=['customerid', 'p_banner'], how='left') \
                               .join(earliest_transaction_df, on=['customerid', 'p_banner'], how='left') \
                               .filter(col('earliest_transaction_date') <= col('start_date')) \
                               .withColumn('epsilon_id', md5(upper(trim(coalesce((col('customerid').cast(StringType()))))))) \
                               .select(col('customerid').alias("customer_id"), 'epsilon_id', 'p_banner', 'start_date', 'end_date', 'deal_yr', 'deal_wk',
                                           'weeks_views', 'shop_view_weeks', 'lambda_views').cache()
        
        logger.info("Final result dataframe:")
        result_df.show()
        shoplikelihood_v4_statistic_features_ctr(week_identifier=self.week_identifier).save(df=result_df)
        result_df.unpersist()


if __name__ == '__main__':
    create_statistic_features = CreateStatisticFeatures()
    create_statistic_features.process()


