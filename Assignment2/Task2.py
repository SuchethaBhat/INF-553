from pyspark import SparkConf, SparkContext
import itertools
from itertools import combinations
from itertools import count
from collections import defaultdict
import sys
import time
import math

start = time.time()

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[5]"))

threshold = int(sys.argv[1])
support = int(sys.argv[2])
input_file = sys.argv[3]
output_file = sys.argv[4]


def to_list(a):
    return [a]


def append(a, b):
    a.append(b)
    return a


def extend(a, b):
    a.extend(b)
    return a


def apriori_items(partition_rdd, actual_support, rdd_length):
    a = list(partition_rdd)
    user_basket_dict = dict(a)
    partition_support = math.ceil(actual_support * len(a) / rdd_length)

    singletons = []
    for i in user_basket_dict:
        singletons.extend(user_basket_dict[i])

    singletons_list = list(dict.fromkeys(singletons))
    single_count = {}

    for i in singletons_list:
        single_count[i] = []
        for j in user_basket_dict:
            if i in user_basket_dict[j]:
                single_count[i].append(j)
            else:
                continue
        if len(single_count[i]) < partition_support:
            del single_count[i]
    freq_items = sorted(list(single_count.keys()))
    final_frequents = []

    def create_combinations(freq_list,count_all):
        comb_result = []
        if type(freq_list[0]) is not tuple:
            pair_combinations = list(combinations(freq_list, 2))
            last = len(pair_combinations)
            comb_result.extend(pair_combinations)
            for pair in range(0,last):
                key = pair_combinations[pair]
                first = pair_combinations[pair][0]
                second = pair_combinations[pair][1]
                value = list(set(count_all[str(first)]) & set(count_all[str(second)]))
                count_all[key] = value

        else:
            i=0
            for i in range(0,2):
                break

            length = len(freq_list[0])
            for i in range(len(freq_list)):
                j = freq_list[i][:(length - 1)]
                m = i + 1
                for z in range(m, len(freq_list)):
                    k = freq_list[z][:(length - 1)]
                    if j == k:
                        tuple_comb = tuple(sorted(set((freq_list[i] + freq_list[z]))))

                        if len(j)==1:
                            k = j[0]

                        count_all[tuple_comb] = list(set(count_all[k]) & set(count_all[tuple_comb[-2]]) &
                                                     set(count_all[tuple_comb[-1]]))
                        comb_result.append(tuple_comb)
                    else:
                        break
        return comb_result,count_all

    def create_freq_itemset(comb, count_all):
        frequents = []
        for pair in comb:
            if len(count_all[pair])>=partition_support:
                frequents.append(pair)
        return frequents

    # user_basket_dict = partition_rdd.collectAsMap()
    # singletons = partition_rdd.map(lambda x: x[1]).reduce(lambda x, y: x + y)

    while True:
        if not freq_items:
            break
        else:
            final_frequents.extend(freq_items)
        combo, count_basket_dic = create_combinations(freq_items,single_count)
        freq_items = create_freq_itemset(combo, count_basket_dic)
    yield final_frequents


def sorting(rdd_value_list):
    singleStrings = []
    tuples = []
    sorted_list = []
    for i in rdd_value_list:
        if type(i) == str:
            singleStrings.append(tuple({i}))
        else:
            tuples.append(i)
    sorted_strings = sorted(singleStrings)
    sorted_tuples = sorted(tuples, key=lambda i: (len(i), i))
    sorted_list = sorted_strings + sorted_tuples
    final_string = ""
    counter = 1
    for i in sorted_list:
        length = len(i)
        if length == 1:
            final_string = final_string + str(i)[:-2] + "),"
        elif length != 1 and length != counter:
            counter = len(i)
            final_string = final_string[:-1] + "\n\n" + str(i) + ","
        else:
            final_string = final_string + str(i) + ","
    return final_string[:-1]


def frequent_son_items(partition, candidates):
    partition_list = list(partition)
    partition_ub_dict = dict(partition_list)
    candi_count = []
    for pair in candidates:
        counter = 0
        for i in partition_ub_dict:
            if type(pair) is tuple:
                if (set(pair)).issubset(tuple(partition_ub_dict[i])):
                    counter += 1
            else:
                if {pair}.issubset(tuple(partition_ub_dict[i])):
                    counter += 1
        candi_count.append((pair, counter))
    return candi_count


small1_rdd = sc.textFile(input_file)
header = small1_rdd.first()

'''
if case_number == 1:
    user_business_baskets = small1_rdd.filter(lambda x: x != header).map(lambda x: x.split(",")).combineByKey(to_list,
                                                                                                              append,
                                                                                                              extend).persist()
else:
    user_business_baskets = small1_rdd.filter(lambda x: x != header).map(lambda x: x.split(",")).map(
        lambda x: (x[1], x[0])).combineByKey(to_list, append, extend).persist()
 
print(user_business_baskets.collect())

'''
user_business_baskets = small1_rdd.filter(lambda x: x != header).\
    map(lambda x: x.split(",")).\
    combineByKey(to_list,append,extend).\
    filter(lambda x: len(list(set(x[1]))) > threshold).persist()

full_rdd_length = user_business_baskets.count()
user_basket_freq_dict = user_business_baskets.collectAsMap()

son_candidate = user_business_baskets.mapPartitions(
    lambda partition: apriori_items(partition, support, full_rdd_length)).flatMap(lambda x: x).map(lambda x: (x, 1)). \
    reduceByKey(lambda x, y: x + y).map(lambda x: x[0]).collect()

son_frequent = user_business_baskets.mapPartitions(lambda x: (frequent_son_items(x, son_candidate))). \
    reduceByKey(lambda x, y: x + y).filter(lambda x: x[1] >= support).map(lambda x: x[0]).collect()
# print(son_frequent)


final_candidate_itemsets = sorting(son_candidate)
# print(final_candidate_itemsets)


final_frequent_itemset = sorting(son_frequent)
# print(final_frequent_itemset)

final_result_string = "Candidates:\n" + final_candidate_itemsets + "\n\n" + "Frequent Itemsets:\n" + final_frequent_itemset
# print(final_result_string)

fp = open(output_file, "w")
fp.write(final_result_string)
fp.close()

print('Duration: ', (time.time() - start))
