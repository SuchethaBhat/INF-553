import sys
import time
from itertools import combinations
import os
from graphframes import GraphFrame
from pyspark import SparkConf, SparkContext
from pyspark.sql import *

start = time.time()

sc = SparkContext.getOrCreate(
    SparkConf().setMaster("local[3]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))
sqlContext = SparkSession.builder.appName('HW4').getOrCreate()
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11  pyspark-shell"
sc.setLogLevel('WARN')


def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


threshold = int(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]


ub_sample = sc.textFile(input_file)
header = ub_sample.first()
ub_sample = ub_sample.filter(lambda x: x != header).map(lambda x: tuple(x.split(","))).distinct().persist()

uid_bid = ub_sample.combineByKey(to_list, append, extend).map(lambda x: (x[0],list(set(x[1])))).filter(
    lambda x: len(list(set(x[1]))) >= threshold).collectAsMap()

uid_pairs = list(combinations(sorted(uid_bid.keys()), 2))

edges = []
vertex_list = []


for i in uid_pairs:
    user1_list = uid_bid[i[0]]
    user2_list = uid_bid[i[1]]
    co_bid = list(set(user1_list) & set(user2_list))
    if len(co_bid) >= threshold:
        edges.append(i)
        edges.append((i[1],i[0]))
        vertex_list.extend([(i[0],), (i[1],)])

vertex_list = list(set(vertex_list))
print(len(vertex_list))
print(len(edges))


vertices_df = sqlContext.createDataFrame(sc.parallelize(vertex_list), ["id"])
edges_df = sqlContext.createDataFrame(sc.parallelize(edges), ["src", "dst"])

g = GraphFrame(vertices_df, edges_df)

community = g.labelPropagation(maxIter=5)
community_list = community.rdd.coalesce(1).map(lambda x: (x[1], x[0])).\
    combineByKey(to_list, append, extend).\
    map(lambda x: (sorted(x[1]))).collect()

final_list = sorted(community_list, key=lambda l: (len(l), l))

with open(output_file, 'w+') as fp:
    for i in final_list:
        fp.writelines(str(i)[1:-1] + "\n")

fp.close()

print("Duration: ", time.time() - start)
