from pyspark import SparkConf, SparkContext
import json
import sys
import time
import math
from itertools import combinations
from collections import Counter

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

input_file = sys.argv[1]
output_file = sys.argv[2]
stopwords_file = sys.argv[3]

s_char = set(r"""([,.!?:;])01"23456789-\&""")

with open(stopwords_file) as f:
    lines = f.read().splitlines()
stopwords_list = [i.lower().strip() for i in lines]


def create_document(rev_text_list, stopwords_list, s_char):
    document = []
    for i in rev_text_list:
        y = i.lower().split()
        for x in y:
            if x is not '':
                z = ''.join(i for i in x if i not in s_char)
                if z not in stopwords_list and z != '':
                    document.append(z)
    return document


def tf_idf_function(review_words_list, bid, totaldocs, ni_dictionary):
    word_count = dict(Counter(review_words_list))
    word_count_list = []
    for i in word_count:
        tuple_count = (i, word_count[i])
        word_count_list.append(tuple_count)
    word_count_list = sorted(word_count_list, key=lambda x: (-x[1], x[0]))
    max_k = word_count_list[0][1]
    tf_idf_list = []
    for i in word_count_list:
        tf = i[1] / max_k
        idf = math.log((totaldocs/ni_dictionary[i[0]]),2)
        tf_idf = tf * idf
        tf_idf_list.append((i[0], tf_idf))

    tf_idf_list = (sorted(tf_idf_list, key=lambda x: (-x[1], x[0]))[:200])
    tf_idf_list_final = []
    for i in tf_idf_list:
        tf_idf_list_final.append(i[0])
    final_tuple = (bid, tf_idf_list_final)
    return final_tuple


def tf_index_bid(index_dictionary, list_userIds, bid):
    index_list = []
    for i in list_userIds:
        index_list.append(index_dictionary[i])
    final = (bid, index_list)
    return final


def replace_bid_with_tfidfwords(bid_tfidflist_dict, bid_list, userid):
    final_list = []
    for i in bid_list:
        final_list.extend(bid_tfidflist_dict[i])
    final_list = dict(Counter(final_list))
    top_words = []
    for i in final_list:
        tuple_count = (i, final_list[i])
        top_words.append(tuple_count)
    top_words = sorted(top_words, key=lambda x: (-x[1], x[0]))[:200]
    final_top_words = []
    for i in top_words:
        final_top_words.append(i[0])
    uid_word_index_tuple = (userid, final_top_words)
    return uid_word_index_tuple


train_review_rdd = sc.textFile(input_file).map(json.loads).\
                        map(lambda x: (x['business_id'], x['text'], x['user_id'])).\
                        distinct().persist()

review_content_rdd = train_review_rdd.map(lambda x: (x[0], x[1])).combineByKey(to_list, append, extend). \
                        map(lambda x: (x[0], create_document(x[1], stopwords_list, s_char))).persist()

N = review_content_rdd.map(lambda x: x[0]).distinct().count()


Ni_dict = review_content_rdd.flatMapValues(lambda x: x).\
            map(lambda x: (x[1], x[0])). \
            combineByKey(to_list, append, extend).\
            map(lambda x: (x[0], len(set(x[1])))).collectAsMap()

business_profile = review_content_rdd.map(lambda x: tf_idf_function(x[1], x[0], N, Ni_dict)).persist()


index_tf_idf_words = business_profile.flatMap(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
bus_profile_replaced_index_dict = business_profile.map(lambda x: tf_index_bid(index_tf_idf_words, x[1], x[0])).collectAsMap()

user_profile = train_review_rdd.map(lambda x: (x[2], x[0])).\
                combineByKey(to_list, append, extend). \
                map(lambda x: (x[0], list(set(x[1])))).\
                map(lambda x: replace_bid_with_tfidfwords(bus_profile_replaced_index_dict, x[1], x[0])).\
                collectAsMap()

result = {'Business_profile':bus_profile_replaced_index_dict,'User_Profile':user_profile}

with open(output_file, 'w') as fp:
    json.dump(result, fp)
fp.close()

print("Duration: ", time.time() - start)
