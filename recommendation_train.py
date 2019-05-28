'''Collaborative Filtering Model Training

    Usage:

    $ spark-submits recommendation_train.py hdfs:/path/to/file.parquet
    hdfs:/path/to/save/model rank regularization alpha

    '''

# We need sys to get the command line arguments
import sys

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS

def main(spark, data_file, model_file, rank, reg, alpha):
    '''Main routine for Collaborative Filtering Model training

        Parameters
        ----------
        spark: SparkSession object

        data_file: string, path to the parquet file to load

        model_file: string, path to store the model

        rank : int, the start/end/step value of dimension of the latent factors

        reg : double, the start/end/step value of regularization parameters

        alpha : double, the _start/_end/_step of scaling parameter for handling implicit feedback (count) data

        '''

    # Load data
    trainIdx = spark.read.parquet(data_file)

    als = ALS(rank=rank, regParam=reg, implicitPrefs=True, alpha=alpha, userCol="user_idx", itemCol="track_idx", ratingCol="count")

    model = als.fit(trainIdx)

    model.write().overwrite().save(model_file)

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('recommendation_train').getOrCreate()

    # Get the filename from the command line
    data_file = sys.argv[1]

    # And the location to store the trained model
    model_file = sys.argv[2]

    rank = int(sys.argv[3])

    reg = float(sys.argv[4])

    alpha = float(sys.argv[5])

    # Call our main routine
    main(spark, data_file, model_file, rank, reg, alpha)
