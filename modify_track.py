'''modify track

    Usage:

    $ spark-submit modify_track.py hdfs:/path/to/train.parquet hdfs:/path/to/train_dropTrack.parquet

'''

# We need sys to get the command line arguments
import sys


from pyspark.sql import SparkSession

def main(spark, train_file, dropTrack_file):
    '''Main routine to transfer identifiers (string) into numerical index

        Parameters
        ----------
        spark: SparkSession object

        string_file: string, path to the parquet file with string identifiers

        index_file: string, path to store the file with numerical indexe
        '''

    # Load data
    cf_train = spark.read.parquet(train_file)
    cf_train.createOrReplaceTempView('cf_train') 
    
    # total counts of each track
    track_sum=spark.sql('select sum(count) as sum, track_id from cf_train group by track_id order by sum asc')
    # select tracks with low total counts
    track_sum.createOrReplaceTempView('track_sum')
    drop1=spark.sql('select track_id, sum from track_sum where sum>5000')
    outcome=cf_train.join(drop1,['track_id'],'leftanti')

    outcome.write.mode('overwrite').parquet(dropTrack_file)

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('modify_track').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    dropTrack_file = sys.argv[2]


    # Call our main routine
    main(spark, train_file, dropTrack_file)
