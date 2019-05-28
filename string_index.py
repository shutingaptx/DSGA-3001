'''String to indexer

    Usage:

    $ spark-submit string_index.py hdfs:/path/to/train.parquet hdfs:/path/to/val.parquet hdfs:/path/to/test.parquet
    hdfs:/path/to/train_index.parquet hdfs:/path/to/val_index.parquet hdfs:/path/to/test_index.parquet

'''

# We need sys to get the command line arguments
import sys


# import pyspark
# sc = pyspark.SparkContext()
# sc.stop()
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

def main(spark, train_file, val_file, test_file, trainIdx_file, valIdx_file, testIdx_file):
    '''Main routine to transfer identifiers (string) into numerical index

        Parameters
        ----------
        spark: SparkSession object

        string_file: string, path to the parquet file with string identifiers

        index_file: string, path to store the file with numerical indexe
        '''

    # Load data
    train = spark.read.parquet(train_file)
    val = spark.read.parquet(val_file)
    test = spark.read.parquet(test_file)

    stringIndexer_user = StringIndexer(inputCol="user_id", outputCol="user_idx")
    stringIndexer_track = StringIndexer(inputCol="track_id", outputCol="track_idx")

    pipeline = Pipeline(stages=[stringIndexer_user,stringIndexer_track])

    stringIndexer = pipeline.fit(train.union(val).union(test))

    trainIdx = stringIndexer.transform(train)
    valIdx = stringIndexer.transform(val)
    testIdx = stringIndexer.transform(test)

    trainIdx.write.mode('overwrite').parquet(trainIdx_file)
    valIdx.write.mode('overwrite').parquet(valIdx_file)
    testIdx.write.mode('overwrite').parquet(testIdx_file)

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('string_index').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    val_file = sys.argv[2]
    test_file = sys.argv[3]

    # And the location to store the trained model
    trainIdx_file = sys.argv[4]
    valIdx_file = sys.argv[5]
    testIdx_file = sys.argv[6]


    # Call our main routine
    main(spark, train_file, val_file, test_file, trainIdx_file, valIdx_file, testIdx_file)
