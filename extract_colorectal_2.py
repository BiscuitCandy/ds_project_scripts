from pymongo import MongoClient
import csv
import sys
import re
import json

# connect to MongoDB and target cord19_%
client = MongoClient('mongodb://readUser:mongo232@bl4:27017/?authSource=test&authMechanism=SCRAM-SHA-256')

db = client['test']
# select the database
CKG1_3 = db['Colorectal_KG']

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
    row_data["table_id"] = document["table_id"]
    row_data["data"] = document["data"] if "data" in document else []
    # write.writerow(row_data)
    if row_data["data"]:
        row_data["data"] = ([" ".join(i).replace("~", "") for i in row_data["data"]])
        print(row_data["id"], "~", row_data["table_id"], "~", row_data["data"])
    idx += 1
    return idx

positives = [
  {
    "$match": {
        
          "tables": {"$ne": "[]"}
        }
    },
    {
            "$unwind": "$tables"
    },
    {
        "$project":
              {
                "title": 1,
                "table_id": "$tables.tableId",
                "data": "$tables.Data"
              }
    }
]

positives_cursor = CKG1_3.aggregate(positives)

row_count=0
with open('./ColorectalKG.csv', 'w') as outfile:
  header = ['id', 'table_id' 'data']
  write = csv.DictWriter(outfile, fieldnames = header ,  delimiter = ',')
  write.writeheader()
  for document in positives_cursor:
    row_count = createDataset(document, write, row_count, positive_condition= True)
    