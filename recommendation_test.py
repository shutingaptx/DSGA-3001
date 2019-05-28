'''Collaborative Filtering Model Testing

    Usage:

    $ spark-submits recommendation_test.py hdfs:/path/to/load/model.parquet
    hdfs:/path/to/testfile K

    '''

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RankingMetrics
import numpy as np
from pyspark.sql import Window
from pyspark.sql.functions import col, expr
import pyspark.sql.functions as F

def main(spark, model_file, data_file, K):
    '''Main routine for Collaborative Filtering Model testing

        Parameters
        ----------
        spark: SparkSession object

        model_file: string, path to store the model

        data_file: string, path to the parquet file to load

        K: int, evaluations are based on predictions of the top K items for each user
        '''
    testIdx = spark.read.parquet(data_file)
    model = ALSModel.load(model_file)

    users_val = testIdx.select("user_idx").distinct()

    perUserPredictedItemsDF = model.recommendForUserSubset(users_val, K)
    perUserPredictedItemsDF = perUserPredictedItemsDF.select("user_idx", "recommendations.track_idx").withColumnRenamed('user_idx', 'user').withColumnRenamed('recommendations.track_idx', 'items')

    w2 = Window.partitionBy('user_idx').orderBy(col('count').desc())
    perUserActualItemsDF = testIdx.select('user_idx', 'track_idx', 'count', F.rank().over(w2).alias('rank')).where('rank <= {0}'.format(K)).groupBy('user_idx').agg(expr('collect_list(track_idx) as items')).withColumnRenamed('user_idx', 'user')

    perUserItemsRDD = perUserPredictedItemsDF.join(perUserActualItemsDF, 'user').rdd.map(lambda row: (row[1], row[2]))
    rankingMetrics = RankingMetrics(perUserItemsRDD)

    print("============================================")
    print("meanAveragePrecision = %.8f" % rankingMetrics.meanAveragePrecision)
    print("precisionAt(K) = %.8f" % rankingMetrics.precisionAt(K))
    print("ndcgAt(K) = %.8f" % rankingMetrics.ndcgAt(K))

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommendation_test').getOrCreate()

    # Get the filename from the command line
    model_file = sys.argv[1]

    # And the location to store the trained model
    data_file = sys.argv[2]

    K = int(sys.argv[3])

    # Call our main routine
    main(spark, model_file, data_file, K)
