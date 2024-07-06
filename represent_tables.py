from pymongo import MongoClient
import csv
import sys
import re
import json

from bs4 import BeautifulSoup


client = MongoClient('mongodb://readUser:mongo232@bl4:27017/?authSource=test&authMechanism=SCRAM-SHA-256')

db = client['test']


colorectal_kg = db['Colorectal_KG']

positives = [
    {
    "$match": {
        "tables.cl_verified": {"$eq": 1}
        }
    },
    {
        "$unwind":"$tables"
    },
    {
    "$project": {
      "_id": 0,
      "table": "$tables.tableData" 
                }
    }
]

def extract_body(table_body) :

    all_rows = table_body.find_all("tr")
    total_rows_len = len(all_rows)

    first_row = table_body.find("tr").find_all("td")
    
    row_len = 0
    for i in first_row :
        row_len += int(i["colspan"].replace('"', '').replace("'", "").replace("/", "").replace("\\", "")) if i.get("colspan") else 1

    table_data = [[0 for i in range(row_len)] for j in range(total_rows_len)]

    # print("*", table_data)

    i, j = 0, 0

    while i < total_rows_len :
        row = all_rows[i].find_all("td")
        # print(row)
        j = 0

        for l in range(row_len):
            if table_data[i][l] == 0 :
                j = l
                break

        for k in row :
            value = k.text 
            row_span = int(k["rowspan"].replace('"', "").replace("'", "").replace("/", "").replace("\\", "")) if k.get("rowspan") else 1
            column_span = int(k["colspan"].replace('"', '').replace("'", "").replace("/", "").replace("\\", "")) if k.get("colspan") else 1

            # print(j, k, value, row_span, column_span)

            for r in range(row_span) :
                b = 0
                while (i+r+b) < total_rows_len and table_data[i+r+b][j] != 0 :
                    # print("########", i, r, b, table_data[i+r+b][j])
                    b += 1
                if (i+r+b) < total_rows_len: 
                    table_data[i + r + b][j] = value
            
            j += 1
            for c in range(column_span -1):
                b = 0
                while (j+c+b) < row_len and table_data[i][j + c + b] != 0 :
                    # print("|||||||||||||", j, c, b, table_data[j+c+b][j])
                    b += 1
                if (j+c+b) < row_len: 
                    table_data[i][j + c + b] = value
        
        i += 1


        # print(table_data)
        # print("*************************************")

    return table_data


def extract_head(table_head, row_len):

    head_rows = table_head.find_all("tr")
    head_rows_len = len(head_rows)

    table_head = [[0 for i in range(row_len)] for j in range(head_rows_len)]
    # print(table_head)

    i = 0

    while i < head_rows_len :
        row = head_rows[i].find_all("td")
        # print(row)
        j = 0

        for l in range(row_len):
            if table_head[i][l] == 0 :
                j = l
                break

        for k in row :
            value = k.text 
            row_span = int(k["rowspan"].replace('"', '').replace("'", "").replace("/", "").replace("\\", "")) if k.get("rowspan") else 1
            column_span = int(k["colspan"].replace('"', '').replace("'", "").replace("/", "").replace("\\", "")) if k.get("colspan") else 1

            # print(j, k, value, row_span, column_span)

            for r in range(row_span) :
                b = 0
                while (i+r+b) < head_rows_len and table_head[i+r+b][j] != 0 :
                    # print("########", i, r, b, table_head[i+r+b][j])
                    b += 1
                if (i+r+b) < head_rows_len: 
                    table_head[i + r + b][j] = value
            
            j += 1
            for c in range(column_span -1):
                b = 0
                while (j+c+b) < row_len and table_head[i][j + c + b] != 0 :
                    # print("|||||||||||||", j, c, b, table_head[j+c+b][j])
                    b += 1
                if (j+c+b) < row_len: 
                    table_head[i][j + c + b] = value
        
        i += 1
        
    head = [0 for i in range(row_len)]

    for i in range(row_len):
        char = ""
        for j in range(len(table_head)) :
            value = str(table_head[j][i])
            char = char if char == value else (char + " : "+  value)
        
        head[i] = char

    return head
    

n = 0
positives_cursor = colorectal_kg.aggregate(positives)
for document in positives_cursor:
    # print(document)

    if not document or document == "{}" :
        continue
    
    html_data = document["table"].replace('\"', '')
    html_data = html_data.replace("<th ", "<td ")
    html_data = html_data.replace("<th>", "<td>")
    html_data = html_data.replace("</th>", "</td>")
    # print(html_data)

    soup = BeautifulSoup(html_data, 'html.parser')
    
    table_body = soup.tbody 
    table_head = soup.thead

    try:
        table_body = extract_body(table_body)
        row_len = len(table_body[0])

        # print("+"*30)
        # print("-"*30)
        # print("+"*30)

        table_head = extract_head(table_head, row_len)

        table_whole = [table_head, *table_body]

        print(table_whole)
        n += 1
    except:
        table_whole = []
        print("A table can not be represented", n+1)
        print("*"*20)
    