from pyspark import SparkConf, SparkContext
import json
import sys
import time
import math
from itertools import combinations
import random

start = time.time()


def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


def create_bid_uidIndex_signature(index_dictionary, user_id_dict, hash_dictionary, list_userIds, bid):
    index_list = []
    bid = user_id_dict[bid]
    for i in list_userIds:
        index_list.append(index_dictionary[i])
    signature_list = []
    for i in hash_dictionary:
        min_hash_list = []
        for j in index_list:
            min_hash_list.append(hash_dictionary[i][j])
        signature_list.append(min(min_hash_list))
    bands = [signature_list[k:k + 1] for k in range(0, len(signature_list), 1)]
    band_tuple = (bid, bands)
    return band_tuple


def pearson_similarity(pair, bid_list_dict, rating_dict):
    try:
        user1_list = bid_list_dict[pair[0]]
        user2_list = bid_list_dict[pair[1]]
        co_rated_bid = list(set(user1_list) & set(user2_list))
        if len(co_rated_bid) >= 3:
            p0_ratings = []
            p1_ratings = []
            for i in co_rated_bid:
                p0_ratings.append(rating_dict[pair[0]][i])
                p1_ratings.append(rating_dict[pair[1]][i])
            p0_avg = sum(p0_ratings) / len(p0_ratings)
            p1_avg = sum(p1_ratings) / len(p1_ratings)

            p0rating_avg = [x - p0_avg for x in p0_ratings]
            p1rating_avg = [y - p1_avg for y in p1_ratings]
            num_list = [p0rating_avg[i] * p1rating_avg[i] for i in range(len(p0rating_avg))]
            numerator = sum(num_list)
            if numerator == 0:
                similarity = -1
            else:
                den_part1 = math.sqrt(sum([x * x for x in p0_ratings]))
                den_part2 = math.sqrt(sum([y * y for y in p1_ratings]))

                denominator = den_part1 * den_part2
                similarity = numerator / denominator
        else:
            similarity = -1
    except:
        similarity = -1
    weight_tuple = (pair, similarity)
    return weight_tuple


def jaccard_pearson(pair, bid_list_dict, user_id_dict, rating_dict):
    user1 = user_id_dict[pair[0]]
    user2 = user_id_dict[pair[1]]

    user1_list = bid_list_dict[user1]
    user2_list = bid_list_dict[user2]
    intersect = list(set(user1_list) & set(user2_list))
    union = list(set(user1_list + user2_list))
    co_rated_len = len(intersect)
    union_len = len(union)
    if co_rated_len >= 3 and (co_rated_len / union_len) >= 0.01:
        p0_ratings = []
        p1_ratings = []
        for i in intersect:
            p0_ratings.append(rating_dict[user1][i])
            p1_ratings.append(rating_dict[user2][i])
        p0_avg = sum(p0_ratings) / len(p0_ratings)
        p1_avg = sum(p1_ratings) / len(p1_ratings)

        p0rating_avg = [x - (p0_avg) for x in p0_ratings]
        p1rating_avg = [y - (p1_avg) for y in p1_ratings]
        num_list = [p0rating_avg[i] * p1rating_avg[i] for i in range(len(p0rating_avg))]
        numerator = sum(num_list)
        if numerator == 0:
            weights_jaccard = ((user1, user2), -1)
            return weights_jaccard
        else:
            den_part1 = math.sqrt(sum([x * x for x in p0_ratings]))
            den_part2 = math.sqrt(sum([y * y for y in p1_ratings]))

            denominator = den_part1 * den_part2
            similarity = numerator / denominator
            if similarity > 0:
                weights_jaccard = ((user1, user2), similarity)
                return weights_jaccard
            else:
                weights_jaccard = ((user1, user2), -1)
                return weights_jaccard
    else:
        weights_jaccard = ((user1, user2), -1)
        return weights_jaccard


sc = SparkContext.getOrCreate(
    SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))

input_file = sys.argv[1]
output_file = sys.argv[2]
case_type = sys.argv[3]

train_review_rdd = sc.textFile(input_file).map(json.loads).map(
    lambda x: (x['business_id'], x['user_id'], x['stars'])).distinct().persist()

if case_type == 'item_based':
    bid_userlist = train_review_rdd.map(lambda x: (x[0], x[1])).combineByKey(to_list, append, extend).persist()
    ratings_list = train_review_rdd.map(lambda x: (x[0], (x[1], x[2]))).combineByKey(to_list, append, extend).map(
        lambda x: (x[0], dict(list(set(x[1]))))).collectAsMap()

    final_bid_userlist = bid_userlist.filter(lambda x: len(list(set(x[1]))) >= 3).collectAsMap()

    print(len(final_bid_userlist.keys()))

    bid_list = list(set(final_bid_userlist.keys()))
    bid_pairs = list(combinations(bid_list, 2))

    final_pairs = []
    for bid_pair in bid_pairs:
        weights = pearson_similarity(bid_pair, final_bid_userlist, ratings_list)

        if weights[1] > 0:
            final_tuple = {"b1": weights[0][0], "b2": weights[0][1], "sim": weights[1]}
            final_pairs.append(final_tuple)
else:

    user_business_rdd = train_review_rdd.map(lambda x: (x[1], x[0])).combineByKey(to_list, append, extend) \
        .filter(lambda x: len(list(set(x[1]))) >= 3).persist()

    bus_rdd_with_index_dic = user_business_rdd.map(lambda x: x[1]).flatMap(
        lambda x: x).distinct().zipWithIndex().collectAsMap()

    user_id_index_dict = user_business_rdd.map(lambda x: x[0]).zipWithIndex().collectAsMap()
    key_index_val_uid_dict = {value: key for key, value in user_id_index_dict.items()}
    user_bidId_dict = user_business_rdd.collectAsMap()
    bid_index_list = list(bus_rdd_with_index_dic.values())

    ratings_list = train_review_rdd.map(lambda x: (x[1], (x[0], x[2]))).combineByKey(to_list, append, extend).map(
        lambda x: (x[0], dict(list(set(x[1]))))).collectAsMap()

    m = len(bid_index_list)
    p = 10259
    primes = [15683, 15791, 15901, 16007, 16103, 16229, 16361, 16453, 16573, 16691, 16787, 16889, 16979, 17041, 17137,
              17231, 17333, 17401, 17483, 17581, 17707, 17827, 17923, 18041, 18127, 18223, 22549, 22699, 23189, 23297,
              23431, 23549, 23789, 24061, 24371, 24181, 27277, 27551, 27739, 27961, 28307, 28843, 28661, 31223, 31667,
              31963, 32771, 32693, 33589, 34019, 35053, 35251, 35731, 36241, 36473, 36901, 71023, 71809, 72337, 68219,
              37171, 37123, 38737, 39461, 39887, 40387, 41413, 43973, 46049, 46399, 46957, 48847, 51197, 51839, 51473,
              51341, 51899, 53681, 54983, 56437, 56827, 58757, 59119, 60089, 60107, 61643, 62137, 64301, 64921, 65881,
              66463, 67211, 67523, 74159, 74507, 74441, 75367, 76481, 78277, 79357]

    hash_dic = {}
    for i in range(1, 41):
        a1 = primes[i - 1]
        a2 = primes[-i]
        b1 = random.choice(primes)
        b2 = random.choice(primes)
        hash_dic[i] = {}
        for j in bid_index_list:
            h1 = (a1 * j + b1) % m
            h2 = (a2 * j + b2) % m
            h3 = ((i * h1 + i * h2 + i * i) % p) % m
            hash_dic[i][j] = h3

    final_pairs = user_business_rdd. \
        map(lambda x: create_bid_uidIndex_signature(bus_rdd_with_index_dic, user_id_index_dict, hash_dic, x[1], x[0])). \
        flatMapValues(lambda x: enumerate(x)). \
        map(lambda x: ((tuple(tuple(x[1][1])), x[1][0]), x[0])). \
        combineByKey(to_list, append, extend). \
        filter(lambda x: len(x[1]) > 1). \
        map(lambda x: list(combinations(sorted(list(set(x[1]))), 2))). \
        flatMap(lambda x: x).distinct(). \
        map(lambda x: jaccard_pearson(x, user_bidId_dict, key_index_val_uid_dict, ratings_list)). \
        filter(lambda x: x[1] > 0).map(lambda x: {"u1": x[0][0], "u2": x[0][1], "sim": x[1]}).collect()

with open(output_file, 'w') as fp:
    for i in final_pairs:
        fp.writelines(json.dumps(i) + "\n")
fp.close()

print("Duration: ", time.time() - start)
