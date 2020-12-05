from pyspark import SparkConf, SparkContext
import json
import sys

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))
review_RDD = sc.textFile(sys.argv[1]).map(json.loads)

directory = {}


def custom_partition(key):
    if key in directory.keys():
        return directory[key]
    else:
        sum = 0
        for i in key:
            sum += ord(i)
        directory[key] = sum
        return sum


partition_type = sys.argv[3]
review_num = int(sys.argv[5])
review_RDD = review_RDD.map(lambda x: (x['business_id'], 1))
if partition_type == 'customized':
    review_RDD = review_RDD.partitionBy(int(sys.argv[4]), custom_partition)

n_partitions = review_RDD.getNumPartitions()
n_items = review_RDD.glom().map(len).collect()
result = review_RDD.reduceByKey(lambda x, y: x + y).filter(lambda x: x[1] > review_num).collect()

final = {"n_partitions": n_partitions, "n_items": n_items, "result": result}
print(final)

with open(sys.argv[2], 'w') as fp:
    json.dump(final, fp)

fp.close()
