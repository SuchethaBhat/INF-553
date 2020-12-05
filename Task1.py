from pyspark import SparkConf, SparkContext
import json
import sys

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

input_file = sys.argv[1]
output_file = sys.argv[2]
stopwords = sys.argv[3]

review_RDD = sc.textFile(input_file).map(json.loads)

'''Output of A - The total number of reviews'''
A = review_RDD.count()

'''Output of B - The number of reviews in a given year, y'''
y = int(sys.argv[4])
B = review_RDD.filter(lambda x: str(y) in x['date']).count()

'''Output of C - The number of distinct users who have written the reviews'''
review_user_id = review_RDD.map(lambda x: x['user_id']).persist()
C = review_user_id.distinct().count()

'''Output of D - Top m users who have the largest number of reviews and its count'''
m = int(sys.argv[5])
D = review_user_id.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1], False).take(m)

'''Output of E - Top n frequent words in the review text. The words should be in lower cases. The following punctuations
“(”, “[”, “,”, “.”, “!”, “?”, “:”, “;”, “]”, “)” and the given stopwords are excluded'''

s_char = set(r"""([,.!?:;])""")
n = int(sys.argv[6])
with open(stopwords) as f:
    lines = f.read().splitlines()
stopwords_list = [i.lower().strip() for i in lines]
print(stopwords_list)


def format_text(rev_text):
    y = rev_text.lower().split(" ")
    j = []
    for x in y:
        if x is not '':
            z = ''.join(i for i in x if i not in s_char)
            if z not in stopwords_list:
                j.append(z)
    return j


E = review_RDD.flatMap(lambda x: format_text(x['text'])). \
    map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b). \
    sortBy(lambda x: x[1], False).map(lambda x: x[0]).take(n)

'''Output to JSON file'''
output_dic = {"A": A, "B": B, "C": C, "D": D, "E": E}
print(output_dic)
with open(output_file, 'w') as fp:
    json.dump(output_dic, fp)

fp.close()
