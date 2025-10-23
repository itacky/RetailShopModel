from pyspark.sql.functions import col, when, lit, rand
from config import params
from app.util.spark.core import SparkMixin


#input
from app.io.sink import shoplikelihood_v4_predictions_final_ctr, shop_likelihood_predicted_data
#output
from app.io.sink import shoplikelihood_v4_test_control_table, shoplikelihood_predicted_final

shop_likelihood_predicted_data.table_prefix = 'prd_v2_'

class MergeShopLikelihoodLabels(SparkMixin):
    def __init__(self):
        super().__init__()
        self.week_identifier = params.WEEK_NUMBER
        self.rand_seed = 42
        self.test_split = 0.1
    

    def process(self):
        v3_predictions = shop_likelihood_predicted_data(week_identifier=self.week_identifier).read(spark=self.spark)
        v4_predictions = shoplikelihood_v4_predictions_final_ctr(week_identifier=self.week_identifier).read(spark=self.spark) \
                                                .join(v3_predictions.select("customer_id").distinct(), on="customer_id", how="inner") \
                                                .withColumn("p_val", rand(self.rand_seed))\
                                                .withColumn("group_flag", when(col("p_val") <=self.test_split, 'T').otherwise(lit('C')))\
                                                .drop("p_val")
                                                
        v4_predictions.groupBy("group_flag").count().show()

        test_control_df = v4_predictions.select("customer_id", "epsilon_id", lit("CTR").alias("banner"), "group_flag")  
        shoplikelihood_v4_test_control_table(week_identifier=self.week_identifier).save(df=test_control_df)
        
        
        test_predictions_df = v4_predictions.filter(col("group_flag") == 'T')\
            .select(col("customer_id"), lit("CTR").alias("banner"), col("shop_prob_label").alias("v4_shop_prob_label"), col("lift_label").alias("v4_lift_label"))\

        adjusted_predictions_df = v3_predictions.join(test_predictions_df, on=['customer_id', 'banner'], how='left') \
            .withColumn("shop_prob_label", when(col("v4_shop_prob_label").isNotNull(), col("v4_shop_prob_label")).otherwise(col("shop_prob_label"))) \
            .withColumn("lift_label", when(col("v4_lift_label").isNotNull(), col("v4_lift_label")).otherwise(col("lift_label"))) \
            .drop("v4_shop_prob_label", "v4_lift_label") \
            .withColumnRenamed("customer_id", "customerid") \
            .distinct()
            
        adjusted_predictions_df.show(20)
        adjusted_predictions_df.groupBy("banner").count().show()
        shoplikelihood_predicted_final(week_identifier=self.week_identifier).save(df=adjusted_predictions_df)


if __name__ == "__main__":
    merge_shoplikelihood_labels = MergeShopLikelihoodLabels()
    merge_shoplikelihood_labels.process()
        
        
        



   