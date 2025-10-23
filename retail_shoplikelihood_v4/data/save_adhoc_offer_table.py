from app.util.spark.core import SparkMixin
from config import params, settings
import logging
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType

#output
from app.io.sink import ad_hoc_offer_info

logger = logging.getLogger(__name__)


class SaveAdHocOfferInfo(SparkMixin):
    def __init__(self):
        self.week_identifier = params.WEEK_NUMBER
        self.sheets_list = [str(x) for x in range(2024, int(self.week_identifier[:4])+1)]
        self.excel_filename = 'ad_hoc_offers_master_tracker.xlsx'
        self.destination_dir = settings.PROJECT_DIR + '/tmp_files/'
        self.destination_filename = self.destination_dir + self.excel_filename
        super().__init__()


    @property
    def ad_hoc_table_schema(self):
        return StructType([
            StructField("offer_description", StringType(), True),
            StructField("start_date", DateType(), True),
            StructField("end_date", DateType(), True),
            StructField("bonus_code", StringType(), True),
            StructField("npp", StringType(), True),
            StructField("funding", StringType(), True),
            StructField("notes", StringType(), True)
        ]) 


    def create_table_from_excel(self, sheets_list):
        ad_hoc_offer_data = []
        for sheet_name in sheets_list:
            sheet_pd = pd.read_excel(self.destination_filename, sheet_name=sheet_name)
            sheet_pd.columns = sheet_pd.columns.str.lower().str.replace(' ', '_')
            parsed_data = sheet_pd[["offer_description", "start_date", "end_date", "bonus_code", "npp", "funding", "notes"]][sheet_pd["npp"].notna()]
            ad_hoc_offer_data.append(parsed_data)

        ad_hoc_offer_pd = pd.concat(ad_hoc_offer_data, ignore_index=True)
        logger.info(ad_hoc_offer_pd.head(10))
        logger.info(ad_hoc_offer_pd.tail(10))
        return ad_hoc_offer_pd


    def process(self):
        ad_hoc_offer_pd = self.create_table_from_excel(self.sheets_list)
        ad_hoc_offer_spark_df = self.spark.createDataFrame(ad_hoc_offer_pd, schema=self.ad_hoc_table_schema)
        ad_hoc_offer_info.save(df=ad_hoc_offer_spark_df)
        logger.info(f"Ad Hoc Offer Info table saved successfully to {ad_hoc_offer_info.table_name}.")



if __name__ == "__main__":
    save_adhoc_offer_info = SaveAdHocOfferInfo()
    save_adhoc_offer_info.process()

