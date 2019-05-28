''' Parameter tuning for Collaborative Filtering model

    Usage:

    $ spark-submit parameter_tuning.py hdfs:/path/to/trainIdx.parquet
     hdfs:/path/to/valIdx.parquet hdfs:/path/to/test.parquet rank_start rank_end rank_step reg_start reg_end reg_step
     alpha_start alpha_end alpha_step K

    '''

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import RankingMetrics
import numpy as np
from pyspark.sql import Window
from pyspark.sql.functions import col, expr
import pyspark.sql.functions as F
import itertools

# def main(spark, train_file, val_file):
def main(spark, trainIdx_file, valIdx_file, rank_start, rank_end, rank_step, reg_start, reg_end, reg_step,
alpha_start, alpha_end, alpha_step, K):
    '''Main routine for parameter tuning

        Parameters
        ----------
        spark : SparkSession object

        train_file : string, path to the parquet file to train

        val_file : string, path to the parquet file to validate

        rank_start/_end/_step : int, the start/end/step value of dimension of the latent factors

        reg_start/_end/_step : double, the start/end/step value of regularization parameters

        alpha_start/_end/_step : double, the _start/_end/_step of scaling parameter for handling implicit feedback (count) data

        K : int, evaluations are based on predictions of the top K items for each user
        '''

    # Load data
    trainIdx = spark.read.parquet(trainIdx_file)
    valIdx = spark.read.parquet(valIdx_file)

    # # Encode the user_id and track_id files as target labels
    # indexerUser = StringIndexer(inputCol="user_id", outputCol="user_idx")
    # indexerTrack = StringIndexer(inputCol="track_id", outputCol="track_idx")
    #
    # pipelineIndexer = Pipeline(stages=[indexerUser,indexerTrack])
    #
    # stringIndexer = pipelineIndexer.fit(train.union(val).union(test))
    # # stringIndexerVal = pipelineIndexer.fit(val)
    #
    # trainIdx = stringIndexer.transform(train)
    # valIdx = stringIndexer.transform(val)
    # testIdx = stringIndexer.transform(test)

    rank_list = np.arange(rank_start, rank_end, rank_step)
    reg_list = np.arange(reg_start, reg_end, reg_step)
    alpha_list = np.arange(alpha_start, alpha_end, alpha_step)

    paramGrid = [rank_list, reg_list, alpha_list]
    users_val = valIdx.select("user_idx").distinct()

    for prm in itertools.product(*paramGrid):
        als = ALS(rank=prm[0], regParam=prm[1], implicitPrefs=True, alpha=prm[2], userCol="user_idx", itemCol="track_idx", ratingCol="count")

        model = als.fit(trainIdx)

        perUserPredictedItemsDF = model.recommendForUserSubset(users_val, K)
        perUserPredictedItemsDF = perUserPredictedItemsDF.select("user_idx", "recommendations.track_idx").withColumnRenamed('user_idx', 'user').withColumnRenamed('recommendations.track_idx', 'items')
        # w1 = Window.partitionBy('user_idx').orderBy(col('prediction').desc())
        # perUserPredictedItemsDF = predictions.select('user_idx', 'track_idx', 'prediction', F.rank().over(w1).alias('rank')).where('rank <= {0}'.format(K)).groupBy('user_idx').agg(expr('collect_list(track_idx) as items')).withColumnRenamed('user_idx', 'user')

        w2 = Window.partitionBy('user_idx').orderBy(col('count').desc())
        perUserActualItemsDF = valIdx.select('user_idx', 'track_idx', 'count', F.rank().over(w2).alias('rank')).where('rank <= {0}'.format(K)).groupBy('user_idx').agg(expr('collect_list(track_idx) as items')).withColumnRenamed('user_idx', 'user')

        perUserItemsRDD = perUserPredictedItemsDF.join(perUserActualItemsDF, 'user').rdd.map(lambda row: (row[1], row[2]))
        rankingMetrics = RankingMetrics(perUserItemsRDD)

        print("============================================")
        print("K: %f; Rank: %f; Regularization: %f; Alpha: %f" %(K, prm[0], prm[1], prm[2]))
        print("meanAveragePrecision = %.8f" % rankingMetrics.meanAveragePrecision)
        print("precisionAt(K) = %.8f" % rankingMetrics.precisionAt(K))
        print("ndcgAt(K) = %.8f" % rankingMetrics.ndcgAt(K))



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('parameter_tuning').getOrCreate()

    # And the location to store the trained model
    trainIdx_file = sys.argv[1]

    # Get the filename from the command line
    valIdx_file = sys.argv[2]

    rank_start = int(sys.argv[3])
    rank_end = int(sys.argv[4])
    rank_step = int(sys.argv[5])

    reg_start = float(sys.argv[6])
    reg_end = float(sys.argv[7])
    reg_step = float(sys.argv[8])

    alpha_start = float(sys.argv[9])
    alpha_end = float(sys.argv[10])
    alpha_step = float(sys.argv[11])

    K = int(sys.argv[12])

    main(spark, trainIdx_file, valIdx_file, rank_start, rank_end, rank_step, reg_start, reg_end, reg_step,
    alpha_start, alpha_end, alpha_step, K)
