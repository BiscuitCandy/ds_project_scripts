from pymongo import MongoClient
import csv
import sys
import re
import json

# connect to MongoDB and target cord19_%
client = MongoClient('mongodb://readUser:mongo232@bl4:27017/?authSource=test&authMechanism=SCRAM-SHA-256')

db = client['test']
# select the database
CKG1_3 = db['CKG1_3']

def check_child(meta_data):
    try:
        value = ""
        for record in meta_data:
            value = record["text"]
            if "children" in record and record["children"]:
                value += " " + check_child(record["children"])
        return value
    except Exception as e:
        print(e)

def createDataset(document, write, idx, positive_condition):
    row_data = {}
    row_data["id"] = document["_id"]
    row_data["caption"] = document["caption"]
    row_data["meta_h"] = check_child(document["meta_h"]) if "meta_h" in document else ""
    row_data["meta_v"] = check_child(document["meta_v"]) if "meta_v" in document else ""
    row_data["data"] = document["data"] if "data" in document else []
    row_data["label"] = 1 if positive_condition else 0
    write.writerow(row_data)
    idx += 1
    return idx

positives = [
  {
    "$match": {
      "$and": [
        {
          "tables": {"$ne": []}
        },
        {
          "$or": [
            {"title": {"$regex": "covid", "$options": "i"}},
            {"title": {"$regex": "corona", "$options": "i"}}
          ]
        },
        {
          "$or": [
            {"tables.tableCaption": {"$regex": "kidney disease", "$options": "i"}},
            {"tables.tableCaption": {"$regex": "renal disease", "$options": "i"}},
            {"tables.tableCaption": {"$regex": "CKD"}}
          ]
        }
      ]
    }
  },
  {
    "$unwind": "$tables"
  },
  {
    "$project":
              {
                  "title": 1,
                "caption": "$tables.tableCaption",
                "meta_h" : "$tables.HMD",
                "meta_v": "$tables.VMD",
                "data": "$tables.Data"
              }
  }
]

positives_cursor = CKG1_3.aggregate(positives)

row_count=0
with open('./positive_query2_data.csv', 'w') as outfile:
  header = ['id', 'caption', 'meta_h', 'meta_v', 'data', 'label']
  write = csv.DictWriter(outfile, fieldnames = header ,  delimiter = ',')
  write.writeheader()
  for document in positives_cursor:
    row_count = createDataset(document, write, row_count, positive_condition= True)

print(f'Total rows in positive set:\t{row_count}')

#########################################
#######################################
negatives = [
  {
    "$match": {
      "$and": [
        {"tables": {"$ne": []}},
        {
          "$and": [
            {"title": {"$not": {"$regex": "covid", "$options": "i"}}},
            {"title": {"$not": {"$regex": "corona", "$options": "i"}}}
          ]
        }
      ]
    }
  },
  {
    "$unwind": "$tables"
  },
  {
    "$project": {
      "title": 1,
      "caption": "$tables.tableCaption",
      "meta_h": "$tables.HMD",
      "meta_v": "$tables.VMD",
      "data": "$tables.Data"
    }
  }, {"$limit": 1017}
]
########################################
##########################################

negatives_cursor = CKG1_3.aggregate(negatives)

row_count=0
with open('./negative_query2_data.csv', 'w') as outfile:
  header = ['id', 'caption', 'meta_h', 'meta_v', 'data', 'label']
  write = csv.DictWriter(outfile, fieldnames = header ,  delimiter = ',')
  write.writeheader()

  for document in negatives_cursor:
    row_count = createDataset(document, write, row_count, positive_condition= False)
print("completed", row_count)