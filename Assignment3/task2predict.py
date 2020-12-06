from pyspark import SparkConf, SparkContext
import json
import sys
import time
import math

start = time.time()


def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


sc = SparkContext.getOrCreate(
    SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))

test_file = sys.argv[1]
model_file = sys.argv[2]
output_file = sys.argv[3]


def cosine_similarity(pair, user, business):
    try:
        user_id = user[pair[0]]
        # print(user_id)
        business_id = business[pair[1]]
        # print(business_id)
        dot_product = len(list(set(user_id) & set(business_id)))
        vector_length_product = math.sqrt(len(user_id)) * math.sqrt(len(business_id))
        cos_sim = dot_product / vector_length_product
        result = (pair, cos_sim)
        return result
    except:
        result = (pair, 0.0000000)
        return result


business_profile = sc.textFile(model_file).map(json.loads).map(lambda x: x['Business_profile']).flatMap(
    lambda x: x.items()).collectAsMap()

user_profile = sc.textFile(model_file).map(json.loads).map(lambda x: x['User_Profile']).flatMap(
    lambda x: x.items()).collectAsMap()

cosine_rdd = sc.textFile(test_file).map(json.loads).map(lambda x: (x['user_id'], x['business_id'])). \
    map(lambda x: cosine_similarity(x, user_profile, business_profile)).filter(lambda x: x[1] >= 0.01).\
    map(lambda x: {"user_id":x[0][0],"business_id":x[0][1],"sim":x[1]}).collect()

with open(output_file, 'w') as fp:
    for i in cosine_rdd:
        fp.writelines(json.dumps(i) + "\n")
fp.close()

print("Duration: ", time.time() - start)
