from pyspark.sql.functions import col, when, lit, exp
from config import params
from app.util.bigquery.connection import SparkBQ


#input
from app.io.sink import shoplikelihood_v4_statistic_features_ctr, shoplikelihood_v4_predictions_ctr, shoplikelihood_v4_test_model_performance

#output
from app.io.sink import shoplikelihood_v4_predictions_final_ctr


class CreateLikelihoodLabels(SparkBQ):
    def __init__(self):
        super().__init__()
        self.week_identifier = params.WEEK_NUMBER
        self.tempLocation = 'temp'

    def get_model_preictions(self):
        bq_table_name = shoplikelihood_v4_predictions_ctr(week_identifier=self.week_identifier).resolve_bq_table_name

        query = f"""
                    Select epsilon_id, epcl_profileid, predicted_purchase_prob
                    From {bq_table_name}

                 """
        print(f"Running query: {query}")
        
        return self.read_from_bq(
            path_to_bq_table=shoplikelihood_v4_predictions_ctr.bq_table_path,
            bq_query=query,
        )
    
    def get_optimal_threshold(self):
        bq_table_name = shoplikelihood_v4_test_model_performance.bq_table_path
        query = f""" Select train_timestamp, optimal_threshold, optimal_f1 from {bq_table_name} where banner = 'CTR' order by train_timestamp desc limit 1 """
        result = self.read_from_bq(
            path_to_bq_table=shoplikelihood_v4_test_model_performance.bq_table_path,
            bq_query=query,
        ).collect()
        train_timestamp = result[0]['train_timestamp']
        optimal_threshold = result[0]['optimal_threshold']
        optimal_f1 = result[0]['optimal_f1']
        print(f"Optimal threshold from model performance table for CTR banner is {optimal_threshold}, f1 score is {optimal_f1} and train timestamp is {train_timestamp}")
        return optimal_threshold


    def get_statistic_features(self):
        return shoplikelihood_v4_statistic_features_ctr(week_identifier=self.week_identifier).read(spark=self.spark)
    

    def process(self):
        model_predictions_df = self.get_model_preictions().drop('epcl_profileid')
        statistics_df = self.get_statistic_features()
        optimal_threshold = self.get_optimal_threshold()

        # Join model predictions with statistics
        joined_df = model_predictions_df.join(statistics_df, on='epsilon_id', how="left")
        joined_df.show(20)
        processed_df = joined_df.withColumn('shop_prob_label', when(col('predicted_purchase_prob') >= optimal_threshold, lit('H')).otherwise(lit('L'))) \
                                .withColumn('prob_shop_n_view_exp_01w', when(col('weeks_views') == 0, lit(0)) \
                                            .otherwise(1 - (exp(-((col('shop_view_weeks') / 52) * 1))))) \
                                .withColumn('prob_view_exp_01w', 1 - (exp(-(col('lambda_views') * 1)))) \
                                .withColumn('aided_prob', (col('prob_shop_n_view_exp_01w') / col('prob_view_exp_01w'))) \
                                .withColumn('lift_ratio', col('aided_prob') / col('predicted_purchase_prob')) \
                                .withColumn('lift_label', when(col('lift_ratio') >= 1.07, lit('H')).otherwise(lit('L'))) \
                                .select('customer_id', 'epsilon_id', 'predicted_purchase_prob', 'prob_shop_n_view_exp_01w', 'prob_view_exp_01w',
                                        'aided_prob', 'shop_prob_label', 'lift_ratio', 'lift_label')

        processed_df.show(100)
        shoplikelihood_v4_predictions_final_ctr(week_identifier=self.week_identifier).save(df=processed_df)


        print(f"Processed {processed_df.count()} records and saved to shoplikelihood_v4_predictions_final_ctr table.")


if __name__ == "__main__":
    create_likelihood_labels = CreateLikelihoodLabels()
    create_likelihood_labels.process()