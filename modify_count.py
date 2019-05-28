'''Modify count

    Usage:

    $ spark-submit modify_count.py hdfs:/path/to/train.parquet hdfs:/path/to/train_drop1Partial.parquet

'''

# We need sys to get the command line arguments
import sys


from pyspark.sql import SparkSession

def main(spark, train_file, drop1Partial_file):

    # Load data
    cf_train = spark.read.parquet(train_file)
    
    # drop all the records with count=1 
#     drop1_all = cf_train.filter('count not in (1)')
    
    # drop first 1M user's records with count=1
    ids = cf_train.select('user_id').distinct()
    top = ids.limit(1000000)#select first 1M users 
    top_cf = cf_train.join(top,['user_id'],'leftsemi')#select first 1M users' records
    top_drop1 = top_cf.filter('count not in (1)')#drop records with count=1
    missing = cf_train.join(top,['user_id'],'leftanti')#select records of users with partial histories
    drop1_partial = top_drop1.union(missing)# union two parts of users
    
    drop1_partial.write.mode('overwrite').parquet(drop1Partial_file)
    

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('modify_count').getOrCreate()

    # Get the filename from the command line
    train_file = sys.argv[1]
    drop1Partial_file = sys.argv[2]


    # Call our main routine
    main(spark, train_file, drop1Partial_file)
