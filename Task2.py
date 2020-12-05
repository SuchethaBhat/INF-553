import json
import sys
import itertools

if sys.argv[4] == 'spark':
    from pyspark import SparkConf, SparkContext
    sc = SparkContext.getOrCreate(SparkConf().setMaster("local[5]"))

    def to_list(a):
        return [a]


    def append(a, b):
        a.append(b)
        return a


    def extend(a, b):
        a.extend(b)
        return a


    review_RDD = sc.textFile(sys.argv[1])
    review_RDD = review_RDD.map(lambda x: json.loads(x))
    review_RDD = review_RDD.map(lambda x: (x['business_id'], float(x['stars']))).combineByKey(to_list, append, extend)

    business_RDD = sc.textFile(sys.argv[2])
    business_RDD = business_RDD.map(lambda x: json.loads(x))
    business_RDD = business_RDD.map(lambda x: (x['business_id'], x['categories'])).filter(lambda x: x[1] is not None) \
        .map(lambda x: (x[0], x[1].split(','))).flatMap(lambda x: [(x[0], i.strip()) for i in x[1]])

    Final = review_RDD.join(business_RDD).map(lambda x: (x[1][1], x[1][0])).combineByKey(to_list, append, extend).map(
        lambda x: (x[0], list(itertools.chain.from_iterable(x[1])))) \
        .map(lambda x: (x[0], (sum(x[1]) / len(x[1])))).sortBy(lambda x: x[0]) \
        .sortBy(lambda x: x[1], False).take(int(sys.argv[5]))
    output_dic = {"result": Final}
    print(output_dic)
    with open(sys.argv[3], 'w') as fp:
        json.dump(output_dic, fp)
    fp.close()

elif sys.argv[4] == 'no_spark':
    from operator import itemgetter
    from collections import defaultdict

    review, business, Business_id_stars, average, category_bid_tuple, Final, cat_star = [], [], [], [], [], [], []
    category_stars = {}

    ''' Load contents from reveiw.json file'''
    f = open(sys.argv[1], 'r')
    for i in f:
        review.append(json.loads(i))

    ''' Load contents from business.json file'''
    b = open(sys.argv[2], 'r', encoding="utf8")
    for j in b:
        business.append(json.loads(j))

    '''extract business_id and stars from review list'''
    for i in review:
        if i['business_id'] is not None:
            Business_id_stars.append((i['business_id'], i['stars']))

    '''extract business_id and categories from business list'''
    for i in business:
        if i['categories'] is not None:
            for j in i['categories'].split(','):
                category_bid_tuple.append((j.strip(), i['business_id']))


    def convert_listoftuple_dictionary(listname, dictname):
        for i, j in listname:
            dictname[i].append(j)


    business_stars = defaultdict(list)
    convert_listoftuple_dictionary(Business_id_stars, business_stars)

    for i in category_bid_tuple:
        if i[1] in business_stars:
            cat_star.append((i[0], business_stars[i[1]]))

    categeory_listofstars = defaultdict(list)
    convert_listoftuple_dictionary(cat_star, categeory_listofstars)

    for i in categeory_listofstars:
        category_stars[i] = list(itertools.chain.from_iterable(categeory_listofstars[i]))

    '''Calculating average for each category'''
    for i in category_stars:
        avg = sum(category_stars[i]) / len(category_stars[i])
        average.append([i, avg])

    average = sorted(average, key=itemgetter(0), reverse=False)
    Final = sorted(average, key=itemgetter(1), reverse=True)

    output_dic = {"result": Final[:int(sys.argv[5])]}
    print(output_dic)
    with open(sys.argv[3], 'w') as fp:
        json.dump(output_dic, fp)

fp.close()
