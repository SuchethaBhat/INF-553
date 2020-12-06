from pyspark import SparkConf, SparkContext
import json
import sys
import time
from itertools import combinations

start = time.time()


def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


def create_bid_uidIndex(index_dictionary, list_userIds, bid):
    index_list = []
    for i in list_userIds:
        index_list.append(index_dictionary[i])
    final = (bid, index_list)
    return final


def create_signature_matrix(hash_dictionary, bid, uid_index_list):
    signature_list = []
    for i in hash_dictionary:
        min_hash_list = []
        for j in uid_index_list:
            min_hash_list.append(hash_dictionary[i][j])
        signature_list.append(min(min_hash_list))
    signature_tuple = (bid, signature_list)
    return signature_tuple


sc = SparkContext.getOrCreate(
    SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))

input_file = sys.argv[1]
output_file = sys.argv[2]

train_review_rdd = sc.textFile(input_file).map(json.loads).map(
    lambda x: (x['business_id'], x['user_id'])).distinct().persist()

user_rdd_with_index = train_review_rdd.map(lambda x: x[1]).distinct().zipWithIndex().persist()
user_rdd_with_index_dic = user_rdd_with_index.collectAsMap()

bid_user_idgroups_rdd = train_review_rdd.combineByKey(to_list, append, extend)
bid_userdId_dict = bid_user_idgroups_rdd.collectAsMap()
bid_user_idgroups_rdd = bid_user_idgroups_rdd.map(lambda x: create_bid_uidIndex(user_rdd_with_index_dic, x[1], x[0]))
user_index_list = list(user_rdd_with_index_dic.values())

m = len(user_index_list)
p = 26189
hash_dic = {}
for i in range(50):
    hash_dic[i] = {}
    for j in user_index_list:
        h1 = (5 * j + 2) % m
        h2 = ((3 * j + 3) % p) % m
        h3 = ((i * h1 + i * h2 + i * i) % p) % m
        hash_dic[i][j] = h3

signature_rdd = bid_user_idgroups_rdd.map(lambda x: create_signature_matrix(hash_dic, x[0], x[1])).map(
    lambda x: (x[0], [x[1][i:i + 1] for i in range(0, len(x[1]), 1)]))

lsh_rdd = signature_rdd.flatMapValues(lambda x: enumerate(x)).map(lambda x: ((tuple(tuple(x[1][1])), x[1][0]), x[0]))

band_rdd = lsh_rdd.combineByKey(to_list, append, extend).filter(lambda x: len(x[1]) > 1).map(
    lambda x: list(combinations(x[1], 2))).flatMap(lambda x: x).distinct()

jaccard_verification = band_rdd.map(lambda x: (
    x, (len(list(set(bid_userdId_dict[x[0]]) & set(bid_userdId_dict[x[1]]))) / len(
        list(set(bid_userdId_dict[x[0]] + bid_userdId_dict[x[1]])))))). \
    filter(lambda x: x[1] >= 0.05).map(lambda x: {"b1": x[0][0], "b2": x[0][1], "sim": x[1]}).collect()



with open(output_file, 'w') as fp:
    for i in jaccard_verification:
        fp.writelines(json.dumps(i) + "\n")
fp.close()



print('Duration: ', time.time() - start)
