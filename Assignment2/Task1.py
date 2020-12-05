from pyspark import SparkConf, SparkContext
from itertools import combinations
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


def apriori_items(partition_rdd, actual_support, rdd_length):
    a = list(partition_rdd)
    user_basket_dict = dict(a)

    partition_support = math.ceil(actual_support * len(a) / (rdd_length))

    singletons = []
    for i in user_basket_dict:
        singletons.extend(user_basket_dict[i])

    singletons_list = list(dict.fromkeys(singletons))
    single_count = {}

    for i in singletons_list:
        single_count[i] = 0
        for j in user_basket_dict:
            if i in user_basket_dict[j]:
                single_count[i] += 1
        if single_count[i] < partition_support:
            del single_count[i]
    freq_items = sorted(list(single_count.keys()))

    final_frequents = []
    comb_counter = 1

    def create_single_from_tuple(keys_list):
        tuple1 = ()
        if type(keys_list[0]) == tuple:
            for i in keys_list:
                tuple1 = tuple1 + i
            List_for_comb = list(set(tuple1))
        else:
            List_for_comb = keys_list
        all_combi_list = sorted(List_for_comb)
        return all_combi_list

    def create_combinations(single_list, counter):
        # sorted_singles = single_list.sort()
        counter = counter + 1
        comb = combinations(single_list, counter)
        comb_result = [comb, counter]
        return comb_result

    def create_freq_itemset(comb, user_basket):
        pair_dic = {}
        for pair in comb:
            pair_dic[pair] = 0
            for j in user_basket:
                if (set(pair)).issubset(tuple(user_basket[j])):
                    pair_dic[pair] += 1
                    if pair_dic[pair] >= partition_support:
                        break
            if pair_dic[pair] < partition_support:
                del (pair_dic[pair])
        frequents = list(dict.fromkeys(pair_dic))
        return frequents

    while True:
        if not freq_items:
            break
        else:
            final_frequents.extend(freq_items)

        for_comb = create_single_from_tuple(freq_items)

        real_comb = create_combinations(for_comb, comb_counter)
        combo = list(real_comb[0])
        comb_counter = real_comb[1]
        freq_items = create_freq_itemset(combo, user_basket_dict)
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


def main():
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    small1_rdd = sc.textFile(input_file)
    header = small1_rdd.first()
    small1_rdd = small1_rdd.filter(lambda x: x != header).map(lambda x: x.split(","))

    if case_number == 1:
        user_business_baskets = small1_rdd.combineByKey(to_list, append, extend).persist()
    else:
        user_business_baskets = small1_rdd.map(lambda x: (x[1], x[0])).combineByKey(to_list, append, extend).persist()

    full_rdd_length = user_business_baskets.count()

    son_candidate = user_business_baskets.mapPartitions(
        lambda partition: apriori_items(partition, support, full_rdd_length)).flatMap(lambda x: x).map(
        lambda x: (x, 1)). \
        reduceByKey(lambda x, y: x + y).map(lambda x: x[0]).collect()

    son_frequent = user_business_baskets.mapPartitions(lambda x: (frequent_son_items(x, son_candidate))). \
        reduceByKey(lambda x, y: x + y).filter(lambda x: x[1] >= support).map(lambda x: x[0]).collect()

    final_candidate_itemsets = sorting(son_candidate)

    final_frequent_itemset = sorting(son_frequent)

    final_result_string = "Candidates:\n" + final_candidate_itemsets + "\n\n" + "Frequent Itemsets:\n" + final_frequent_itemset

    fp = open(output_file, "w")
    fp.write(final_result_string)
    fp.close()


if __name__ == "__main__":
    main()

'''
son_frequent = son_candidate.map(lambda x: (x, frequent_son_items(x, user_basket_freq_dict,support))).filter(
    lambda x: x[1] >= support).map(lambda x: x[0]).collect()
'''

print('Duration: ', (time.time() - start))
