from pyspark.sql import DataFrame
from pyspark.sql.functions import lit, row_number, col, concat, lpad, md5, upper, trim, coalesce, when
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window
from app.util.spark.core import SparkMixin
import datetime
import logging
import sys
import argparse

from config import params

#input
from app.io.source import transaction_detail, transaction_summary, loyalty_account_current, reco_week_reference


#output
from app.io.sink import shoplikelihood_v4_customer_purchase_profile_ctr, shoplikelihood_v4_customer_point_profile_ctr, shoplikelihood_v4_customer_purchase_history_ctr, shoplikelihood_v4_npp_history_ctr, \
ad_hoc_offer_info

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

window_days = [30, 60, 90, 120, 365]

class PrepareCustomerFeatures(SparkMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = self.get_args()
        self.week_identifier = self.args.week_identifier
        self.npp_history_flag = self.args.npp_history_flag
        self.reco_reference_df = self.load_reco_reference_df()
        self.data_window = [2024, 2025]
        print(f"Creating Customer Features for Week Identifier: {self.week_identifier}")
    
    
    def get_args(self):
        parser = argparse.ArgumentParser(description='Create Customer Features')
        parser.add_argument('--week_identifier', required=True, type=str, help='Week identifier in the format YYYY_3WW')
        parser.add_argument('--npp_history_flag', required=False, type=str, default='N', help='Flag to create NPP history data, Y or N')
        return parser.parse_args()
    

    def load_reco_reference_df(self) -> DataFrame:
        return reco_week_reference.read(spark=self.spark) \
                                  .select("start_date", "end_date", "deal_yr", "deal_wk") \
                                  .withColumn("week_identifier", concat(col("deal_yr"), lit("_3"), lpad(col("deal_wk"),2,'0')))

    def load_transaction_data(self, dateid_begin, dateid_end) -> DataFrame:
        windowSpec = Window.partitionBy("cardnumber", "banner").orderBy("tx_ts")
        sales_window = Window.partitionBy("cardnumber", "banner", "lms_transaction_id")
        return transaction_detail.read(spark=self.spark) \
                                                  .select(col("loyalty_card_num").alias("cardnumber"), "p_trans_dateid","banner", "sales", "lms_transaction_id") \
                                                  .filter(col("p_trans_dateid").between(dateid_begin, dateid_end)) \
                                                  .filter("transactiontype = 'Purchase'") \
                                                  .filter("p_banner = 'CTR'") \
                                                  .filter("cardnumber is not NULL") \
                                                  .withColumn("purchase_flag", lit(1)) \
                                                   .withColumn("sales", F.sum("sales").over(sales_window)) \
                                                  .withColumn("tx_ts", F.to_timestamp("p_trans_dateid", "yyyyMMdd")) \
                                                  .withColumn("tx_epoch", F.col("tx_ts").cast("long")) \
                                                  .distinct()\
                                                  .withColumn("prev_tx_ts", F.lag("tx_ts").over(windowSpec)) \
                                                  .withColumn("current_date", F.to_timestamp(F.current_date())) \
    
    def load_transaction_summary_data(self, dateid_begin, dateid_end) -> DataFrame:
        windowSpec = Window.partitionBy("cardnumber", "banner").orderBy("tx_ts")
        return transaction_summary.read(spark=self.spark) \
                                  .select("cardnumber", "banner", "loyalty_base_points_dol", "loyalty_credit_points_dol",
                                                          "loyalty_bonus_points_dol", "loyalty_redemption_points_dol", "loyalty_total_points_dol", 
                                                          (col("dollarbalance") * 0.01).alias("dollarbalance"),  "enrolldate", "registerdate", "p_trans_dateid")\
                                  .filter(col("p_trans_dateid").between(dateid_begin, dateid_end)) \
                                  .filter("p_banner = 'CTR'") \
                                  .withColumn("tx_ts", F.to_timestamp("p_trans_dateid", "yyyyMMdd")) \
                                  .withColumn("tx_epoch", F.col("tx_ts").cast("long")) \
                                  .withColumn("prev_tx_ts", F.lag("tx_ts").over(windowSpec)) \
                                  .withColumn("current_date", F.to_timestamp(F.current_date()))
                        
    def create_binary_purchase_cols_for_N_weeks(self, transaction_detail_df: DataFrame, weeks: int) -> DataFrame:
        end_date = self.reco_reference_df.filter(col("week_identifier") == self.week_identifier).first().end_date
        start_date = self.reco_reference_df.filter(col("week_identifier") == self.week_identifier).first().start_date - datetime.timedelta(weeks=weeks)  # Calculate start date based on weeks

        reference_df = self.reco_reference_df.filter((col("start_date") >= start_date) & (col("end_date") <= end_date)) \
                                             .select("week_identifier", "start_date", "end_date") \
                                             .withColumn("start_date", F.date_format("start_date", "yyyyMMdd")) \
                                             .withColumn("end_date", F.date_format("end_date", "yyyyMMdd"))

        transactions_with_week = transaction_detail_df.join(reference_df,
                                                            (transaction_detail_df.p_trans_dateid >= reference_df.start_date) &
                                                            (transaction_detail_df.p_trans_dateid <= reference_df.end_date),
                                                            how="inner")
        
        customer_purchase_history = transactions_with_week.groupBy("cardnumber", "banner", "week_identifier") \
                                               .agg(F.lit(1).alias("made_purchase"))
        
        customer_purchase_history = customer_purchase_history.fillna(0).cache()
        customer_purchase_history.show()

        return customer_purchase_history
    
    def create_npp_history_df(self) -> DataFrame:

        starting_year = min(self.data_window)
        ending_year = max(self.data_window)
        adhoc_offers_df = ad_hoc_offer_info.read(spark=self.spark)
        logger.info(f"Displaying Adhoc Offers DataFrame:")
        adhoc_offers_df.show()
        
        campaign_dates_df = adhoc_offers_df.withColumn("campaign_day", F.explode(F.sequence(col("start_date"), col("end_date"))))
        joined_df = campaign_dates_df.join(
                                            self.reco_reference_df,
                                            (campaign_dates_df.campaign_day >= self.reco_reference_df.start_date) &
                                            (campaign_dates_df.campaign_day <= self.reco_reference_df.end_date),
                                            how="inner"
                                        ).drop(self.reco_reference_df.start_date)\
                                        .drop(self.reco_reference_df.end_date)
        
        week_identifier_df = self.reco_reference_df.filter(col("deal_yr").between(starting_year, ending_year)) \
                                                   .select("week_identifier").distinct()
        
        final_df = joined_df.drop("campaign_day").distinct() \
                            .filter(col("npp") == 'Y') \
                            .filter(col("funding").like('%MGF%')) \
                            .filter(F.upper(col("offer_description")).like("%CTR%")) \
                            .filter("upper(offer_description) NOT LIKE '%PARTY CITY%'") \
                            .filter("upper(offer_description) NOT LIKE '%PC%'")\
                            .filter(col("deal_yr").between(starting_year, ending_year))
        
        result_df = week_identifier_df.join(final_df.select("week_identifier", lit(1).alias("npp_flag")).distinct(), on='week_identifier', how='left') \
                                      .withColumn("deal_yr", week_identifier_df.week_identifier[0:4].cast("int")) \
                                      .withColumn("deal_wk", week_identifier_df.week_identifier[6:2].cast("int")) \
                                      .withColumn("npp_flag", when(col("npp_flag").isNull(), lit(0)).otherwise(col("npp_flag")))\
                                      .distinct()
                                      
        result_df.groupBy("deal_yr").sum("npp_flag").show()
        return result_df.select("week_identifier", "deal_yr", "deal_wk", "npp_flag").cache()
    

    def process(self):
        loyalty_account_df = loyalty_account_current.read(spark=self.spark)\
                                                    .withColumn('epsilon_id', md5(upper(trim(coalesce((col('customerid').cast(StringType()))))))) \
                                                    .select("epcl_profileid", "epsilon_id", "cardnumber")

        # get the date range for the last 3 years
        end_date = self.reco_reference_df.filter(col("week_identifier") == self.week_identifier).first().start_date
        start_date = end_date - datetime.timedelta(days=1095)
        date_begin = start_date.strftime("%Y%m%d")
        date_end = end_date.strftime("%Y%m%d")

        # load transaction data
        transaction_detail_df = self.load_transaction_data(dateid_begin=date_begin, dateid_end=date_end).cache()
        transaction_summary_df = self.load_transaction_summary_data(dateid_begin=date_begin, dateid_end=date_end).cache()

        #create df with recent dollar balance
        window_spec = Window.partitionBy("cardnumber", "banner").orderBy(col("p_trans_dateid").desc())
        recent_dollar_df = transaction_summary_df.withColumn("rn", row_number().over(window_spec)) \
                            .filter(col("rn") == 1) \
                            .select("cardnumber", "banner", "dollarbalance").distinct()

        # create customer purchase/point profile dfs
        customer_details_flagged_df = transaction_detail_df.withColumn("days_since_tx", F.datediff(col("current_date"), col("tx_ts"))) \
                                                 .withColumn("days_since_last_purchase", F.datediff("tx_ts", "prev_tx_ts")) 
        
        transaction_summary_flagged_df = transaction_summary_df.withColumn("days_since_tx", F.datediff(col("current_date"), col("tx_ts"))) \
                                                 .withColumn("days_since_last_purchase", F.datediff("tx_ts", "prev_tx_ts")) 
        
        # Build aggregation expressions for purchases and sales
        purchase_profile_agg_exprs = [
        F.count("purchase_flag").alias("total_purchases"),
        F.min("p_trans_dateid").alias("first_purchase_date"),
        F.max("p_trans_dateid").alias("last_purchase_date"),
        F.expr("percentile_approx(days_since_last_purchase, 0.25)").alias("days_since_last_purchase_p25"),
        F.expr("percentile_approx(days_since_last_purchase, 0.50)").alias("days_since_last_purchase_p50"),
        F.expr("percentile_approx(days_since_last_purchase, 0.75)").alias("days_since_last_purchase_p75"),
        F.avg("days_since_last_purchase").alias("avg_days_since_last_purchase"),
        F.stddev("days_since_last_purchase").alias("stddev_days_since_last_purchase"),
        ]

        # Build aggregation expressions for points
        points_profile_agg_exprs = [
        F.max("registerdate").alias("register_date"),
        F.max("enrolldate").alias("enroll_date"),
        ]

        for days in window_days:
            purchase_profile_agg_exprs.append(F.sum(f"tx_in_{days}").alias(f"num_of_purchases_last_{days}_days"))
            purchase_profile_agg_exprs.append(F.sum(f"sales_in_{days}").alias(f"sales_last_{days}_days"))

            points_profile_agg_exprs.append(F.sum(f"base_points_in_{days}").alias(f"base_points_last_{days}_days"))
            points_profile_agg_exprs.append(F.sum(f"credit_points_in_{days}").alias(f"credit_points_last_{days}_days"))
            points_profile_agg_exprs.append(F.sum(f"bonus_points_in_{days}").alias(f"bonus_points_last_{days}_days"))
            points_profile_agg_exprs.append(F.sum(f"redemption_points_in_{days}").alias(f"redemption_points_last_{days}_days"))
            points_profile_agg_exprs.append(F.sum(f"total_points_in_{days}").alias(f"total_loyalty_point_balance_last_{days}_days"))
            
            customer_details_flagged_df = customer_details_flagged_df.withColumn(f"tx_in_{days}", when(col("days_since_tx") <= days, 1).otherwise(0)) \
                                                                     .withColumn(f"sales_in_{days}", when(col("days_since_tx") <= days, col("sales")).otherwise(0.0))
            
            transaction_summary_flagged_df = transaction_summary_flagged_df.withColumn(f"base_points_in_{days}", when(col("days_since_tx") <= days, col("loyalty_base_points_dol")).otherwise(0.0)) \
                                                                           .withColumn(f"credit_points_in_{days}", when(col("days_since_tx") <= days, col("loyalty_credit_points_dol")).otherwise(0.0)) \
                                                                           .withColumn(f"bonus_points_in_{days}", when(col("days_since_tx") <= days, col("loyalty_bonus_points_dol")).otherwise(0.0)) \
                                                                           .withColumn(f"redemption_points_in_{days}", when(col("days_since_tx") <= days, col("loyalty_redemption_points_dol")).otherwise(0.0)) \
                                                                           .withColumn(f"total_points_in_{days}", when(col("days_since_tx") <= days, col("loyalty_total_points_dol")).otherwise(0.0))         

        customer_purchase_profile_df = customer_details_flagged_df.groupBy("cardnumber", "banner").agg(*purchase_profile_agg_exprs)\
                                                                  .withColumn("days_since_first_purchase", F.datediff(F.current_date(), F.to_date(F.col("first_purchase_date"), "yyyyMMdd"))) \
                                                                  .withColumn("days_since_last_purchase", F.datediff(F.current_date(), F.to_date(F.col("last_purchase_date"), "yyyyMMdd"))) \
                                                                  .join(loyalty_account_df, on=["cardnumber"], how="inner") \
                                                                  .drop("cardnumber").cache()
        
        customer_point_profile_df =  transaction_summary_flagged_df.groupBy("cardnumber", "banner").agg(*points_profile_agg_exprs) \
                                                                   .join(recent_dollar_df, on=["cardnumber", "banner"], how="left") \
                                                                   .join(loyalty_account_df, on=["cardnumber"], how="inner") \
                                                                   .drop("cardnumber").cache()


        # create binary purchase columns for the last 52 weeks
        customer_purchase_history = self.create_binary_purchase_cols_for_N_weeks(transaction_detail_df, weeks=104) \
                                        .join(loyalty_account_df, on=["cardnumber"], how="inner") \
                                        .drop("cardnumber").cache()   
        
        if self.npp_history_flag == 'Y':
            logger.info("Creating NPP History DataFrame as npp_history_flag is set to 'Y'") 
            npp_history_df = self.create_npp_history_df()
            npp_history_df.show(50)
        else:
            logger.info("Skipping NPP History DataFrame creation as npp_history_flag is not set to 'Y'") 

        # display the dataframes
        customer_purchase_profile_df.show(50)
        customer_point_profile_df.show(50)
        customer_purchase_history.show(50)
        

        # save the customer features tables
        shoplikelihood_v4_customer_purchase_profile_ctr(week_identifier=self.week_identifier).save(df=customer_purchase_profile_df)
        shoplikelihood_v4_customer_point_profile_ctr(week_identifier=self.week_identifier).save(df=customer_point_profile_df)
        shoplikelihood_v4_customer_purchase_history_ctr(week_identifier=self.week_identifier).save(df=customer_purchase_history)
        if self.npp_history_flag == 'Y':
            shoplikelihood_v4_npp_history_ctr.save(df=npp_history_df)

    
if __name__ == "__main__":
    prepare_customer_features = PrepareCustomerFeatures()
    prepare_customer_features.process()

  
        