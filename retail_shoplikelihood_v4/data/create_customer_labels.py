from pyspark.sql import DataFrame
from pyspark.sql.functions import lit, col, concat, lpad, md5, upper, trim, coalesce
from pyspark.sql.types import StringType
from app.util.spark.core import SparkMixin
import logging
import sys


from config import params

from app.io.source import transaction_detail, loyalty_account_current, reco_week_reference

from app.io.sink import shoplikelihood_v4_customer_purchase_labels_ctr

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class CreateCustomerPurchaseLabels(SparkMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_week_identifier = params.LAST_TWO_WEEK_NUMBER
        logger.info(f"Available Week_identifiers: {params.WEEK_NUMBER}, {params.LAST_WEEK_NUMBER},{params.LAST_TWO_WEEK_NUMBER},{params.LAST_THREE_WEEK_NUMBER}, {params.AFTER_FOUR_WEEKS_NUMBER}")
    
    def load_reco_reference_df(self) -> DataFrame:
        return reco_week_reference.read(spark=self.spark) \
                                  .select("start_date", "end_date", "deal_yr", "deal_wk") \
                                  .withColumn("week_identifier", concat(col("deal_yr"), lit("_3"), lpad(col("deal_wk"),2,'0')))


    def load_transaction_data(self, dateid_begin, dateid_end) -> DataFrame:
        logger.info(f"Loading transaction data between {dateid_begin} and {dateid_end}")
        return transaction_detail.read(spark=self.spark) \
                                                  .select(col("loyalty_card_num").alias("cardnumber"), "banner") \
                                                  .filter(col("p_trans_dateid").between(dateid_begin, dateid_end)) \
                                                  .filter("transactiontype = 'Purchase'") \
                                                  .filter("p_banner = 'CTR'") \
                                                  .filter("cardnumber is not NULL") \
                                                  .withColumn("purchase_flag", lit("Y")) \
                                                  .distinct()

    def process(self):
        print("Preparing customer purchase labels...")

        # Load the week reference data
        reco_reference_df = self.load_reco_reference_df()
        dateid_begin = reco_reference_df.filter(col("week_identifier") == self.target_week_identifier).first().start_date.strftime("%Y%m%d")
        dateid_end = reco_reference_df.filter(col("week_identifier") == self.target_week_identifier).first().end_date.strftime("%Y%m%d")
        transaction_data_df = self.load_transaction_data(dateid_begin, dateid_end)
        transaction_data_df.show()
        loyalty_account_df = loyalty_account_current.read(spark=self.spark)\
                                                    .withColumn('epsilon_id', md5(upper(trim(coalesce((col('customerid').cast(StringType()))))))) \
                                                    .select("epcl_profileid", "epsilon_id", "cardnumber")
        loyalty_account_df.show()

        #create purchase labels for training
        customer_purchase_df = transaction_data_df.join(loyalty_account_df, on='cardnumber', how='inner') \
                                                  .select("epcl_profileid", "epsilon_id", "banner", "purchase_flag")
        customer_purchase_df.show(50)
        print(f"Customer Purchase Count: {customer_purchase_df.count()}")

        # Save the DataFrame to a table        
        shoplikelihood_v4_customer_purchase_labels_ctr(week_identifier=self.target_week_identifier).save(df=customer_purchase_df)


if __name__ == "__main__":
    create_customer_purchase_labels = CreateCustomerPurchaseLabels()
    create_customer_purchase_labels.process()
