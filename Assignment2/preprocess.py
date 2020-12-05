from pyspark import SparkConf, SparkContext
import json
import sys
import csv

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

review_RDD = sc.textFile('review.json').map(json.loads).map(lambda x: (x['user_id'], x['business_id']))
business_RDD = sc.textFile('business.json').map(json.loads).filter(lambda x: x['state'] == 'NV').map(
    lambda x: x['business_id']).collect()

final = review_RDD.filter(lambda x: x[1] in business_RDD).collect()


def create_csv(output_file_path, data):
    with open(output_file_path, 'w', newline='') as fp:
        writer = csv.writer(fp, quoting=csv.QUOTE_NONE)
        writer.writerow(["user_id", "business_id"])
        for row in data:
            writer.writerow(row)
    fp.close()


create_csv('Sample_data.csv', final)
