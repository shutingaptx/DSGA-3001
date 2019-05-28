'''Down sample

    Usage:

    $ spark-submit down_sample.py hdfs:/path/to/train.parquet hdfs:/path/to/train_downsample.parquet

'''

# We need sys to get the command line arguments
import sys


from pyspark.sql import SparkSession

def main(spark, train_file, trainSmall_file):
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
    
    # we have in total 1129318 distinct users 
    ids=spark.sql('select distinct(user_id) from cf_train')
    ids.createOrReplaceTempView('ids')
    
    # select first 1M users
    top=spark.sql('select user_id from ids limit 1000000')
    top.createOrReplaceTempView('top')
    
    # randomly choose 200000 users from 1M users
    regular=spark.sql('select * from top order by rand() limit 200000')
    regular.createOrReplaceTempView('regular')
    
    # getting the 110K users with missing records
    missing=spark.sql('select user_id from ids where not exists (select user_id from top where ids.user_id=top.user_id)')
    missing.createOrReplaceTempView('missing')
    
    # union the chosen users
    down=spark.sql('select * from regular union all select * from missing')
    down.createOrReplaceTempView('down')
    
    #down_cf=spark.sql('select * from cf_train where user_id in (select user_id from down)')
    
    down_cf=spark.sql('select * from cf_train where exists (select user_id from down where cf_train.user_id=down.user_id)')

    down_cf.write.mode('overwrite').parquet(trainSmall_file)

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('down_sample').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    trainSmall_file = sys.argv[2]


    # Call our main routine
    main(spark, train_file, trainSmall_file)
