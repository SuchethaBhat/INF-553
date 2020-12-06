from pyspark import SparkConf, SparkContext
import json
import sys
import time
import math
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


sc = SparkContext.getOrCreate(
    SparkConf().setMaster("local[*]").set("spark.executor.memory", "4g").set("spark.driver.memory", "4g"))

train_review = sys.argv[1]
test_review = sys.argv[2]
model_file = sys.argv[3]
output_file = sys.argv[4]
cf_type = sys.argv[5]
avg_file = sys.argv[6]

train_review_rdd = sc.textFile(train_review).map(json.loads).map(
    lambda x: (x['business_id'], x['user_id'], x['stars'])).distinct()

if cf_type == 'item_based':

    ratings_list = train_review_rdd.map(lambda x: (x[1], (x[0], x[2]))).combineByKey(to_list, append, extend).map(
        lambda x: (x[0], dict(list(set(x[1]))))).collectAsMap()

    model_rdd = sc.textFile(model_file).map(json.loads).map(lambda x: ((x['b1'], x['b2']), x['sim'])).collectAsMap()

    def predict_rating(test_pair, ratings_dictionary, model_dictionary):
        try:
            user_id = test_pair[0]
            business_id = test_pair[1]
            business_list_rated = list(ratings_dictionary[user_id].keys())
            comb_bid = []
            for i in business_list_rated:
                if (business_id, i) in model_dictionary.keys():
                    similarity = model_dictionary[(business_id, i)]
                    pair_similarity = (i, similarity)
                    comb_bid.append(pair_similarity)
                elif (i, business_id,) in model_dictionary.keys():
                    similarity = model_dictionary[(i, business_id)]
                    pair_similarity = (i, similarity)
                    comb_bid.append(pair_similarity)
                else:
                    continue
            comb_bid = sorted(comb_bid, key=lambda x: (-x[1], x[0]))[:3]
            numerator = 0
            denominator = 0
            for j in comb_bid:
                numerator = numerator + ((ratings_dictionary[user_id][j[0]]) * j[1])
                denominator = denominator + j[1]
            predicted_rating = numerator / denominator
            final_value = (test_pair, predicted_rating)
            return final_value
        except:
            predicted_rating = -1
            final_value = (test_pair, predicted_rating)
            return final_value


    test_review_rdd = sc.textFile(test_review).map(json.loads).map(lambda x: (x['user_id'], x['business_id'])).map(
        lambda x: predict_rating(x, ratings_list, model_rdd)).filter(lambda x: x[1] >= 0).map(
        lambda x: {"user_id": x[0][0], "business_id": x[0][1], "stars": x[1]}).collect()


else:

    uid_bid_dict = train_review_rdd.map(lambda x: (x[1],x[0])).combineByKey(to_list, append, extend).\
        map(lambda x: (x[0],list(set(x[1])))).collectAsMap()
    ratings_list = train_review_rdd.map(lambda x: (x[0], (x[1], x[2]))).combineByKey(to_list, append, extend).map(
        lambda x: (x[0], dict(list(set(x[1]))))).collectAsMap()
    model_rdd = sc.textFile(model_file).map(json.loads).map(lambda x: ((x['u1'], x['u2']), x['sim'])).collectAsMap()
    with open('path_to_file/person.json') as f:
        data = json.load(f)


    def predict_rating_userbased(test_pair, ratings_dictionary, model_dictionary,uid_bid_dictionary):
        try:
            user_id = test_pair[0]
            business_id = test_pair[1]
            user_list_rated = list(ratings_dictionary[business_id].keys())
            comb_bid = []
            for i in user_list_rated:
                if (user_id, i) in model_dictionary.keys():
                    similarity = model_dictionary[(user_id, i)]
                    pair_similarity = (i, similarity)
                    comb_bid.append(pair_similarity)
                elif (i, user_id,) in model_dictionary.keys():
                    similarity = model_dictionary[(i, user_id)]
                    pair_similarity = (i, similarity)
                    comb_bid.append(pair_similarity)
                else:
                    continue
            comb_bid = sorted(comb_bid, key=lambda x: (-x[1], x[0]))[:11]

            active_user_bids = uid_bid_dictionary[user_id]
            active_user_rating = []
            for k in active_user_bids:
                active_user_rating.append(ratings_dictionary[k][user_id])
            active_user_avg = sum(active_user_rating) / len(active_user_rating)
            numerator = 0
            denominator = 0
            if len(comb_bid)<3:
                predicted_rating = -1
                final_value = (test_pair, predicted_rating)
                return final_value
            for j in comb_bid:
                user2_bids = uid_bid_dictionary[j[0]]
                intersect = list(set(active_user_bids) & set(user2_bids))
                user2_rating =[]
                for k in intersect:
                    user2_rating.append(ratings_dictionary[k][j[0]])
                user2_avg = sum(user2_rating)/len(user2_rating)
                numerator = numerator + ((ratings_dictionary[business_id][j[0]] - user2_avg) * j[1])
                denominator = denominator + j[1]
            predicted_rating = active_user_avg + (numerator / denominator)
            final_value = (test_pair, predicted_rating)
            return final_value
        except:
            predicted_rating = -1
            final_value = (test_pair, predicted_rating)
            return final_value

    test_review_rdd = sc.textFile(test_review).map(json.loads).map(lambda x: (x['user_id'], x['business_id'])).map(
        lambda x: predict_rating_userbased(x, ratings_list, model_rdd,uid_bid_dict)).filter(lambda x: x[1] >= 0).map(
        lambda x: {"user_id": x[0][0], "business_id": x[0][1], "stars": x[1]}).collect()
    print(len(test_review_rdd))

with open(output_file, 'w') as fp:
    for i in test_review_rdd:
        fp.writelines(json.dumps(i) + "\n")
fp.close()

print("Duration: ", time.time() - start)
