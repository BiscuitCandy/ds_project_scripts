import pandas as pd

import sys
import re
import json
import os
import torch
import numpy as np

from pymongo import MongoClient
from bs4 import BeautifulSoup
from scipy.spatial.distance import cosine
from transformers import BertModel, BertTokenizer

from biobert_embeddings import BioBert # User Defined Class

client = MongoClient('mongodb://readUser:mongo232@bl4:27017/?authSource=test&authMechanism=SCRAM-SHA-256')

db = client['test']

# colorectal_kg = db['CancerKG1_0_1']
colorectal_kg = db['Colorectal_KG']



unit_pattern = {}
units = ['years', 'days', 'ml/h/kg', 'mg*h/l', 'mg/kg', 'hb/g', 'ng/ml', 'ug/ml', 'pg/ml', 'mg/ml', 'mg/kg', 'ug/l', 'kg/m2', 'kg/l', 'kg/m3', 'kg/m2', 'kg/m', 'mg/m3', 'mg/m2', 'mg/m' 'g/m3', 'g/m2', 'g/m' 'pg/l', 'ng/l', 'g/l', 'g l-1', 'g/m', 'ml', 'cm2', 'cm', 'km', 'kg', 'pg', 'mg', 'g', 'l', 'm3', 'm2', 'm']

for i in units :
    pattern = '[A-Za-z0-9]*\s+' + i +'\s+[A-Za-z0-9]*'
    p = re.compile(pattern, re.IGNORECASE)
    unit_pattern[p] = i

def return_units(data):
    for i in unit_pattern :
        if i.search(str(data)) :
            return unit_pattern[i]




def extract_body(table_body) :

    all_rows = table_body.find_all("tr")
    total_rows_len = len(all_rows)

    first_row = table_body.find("tr").find_all("td")

    row_len = 0
    for i in first_row :
        row_len += int(i["colspan"].replace('"', '').replace("'", "").replace("/", "").replace("\\", "")) if i.get("colspan") else 1

    table_data = [[0 for i in range(row_len)] for j in range(total_rows_len)]

    for i in range(len(all_rows)) :

        j = 0

        for k in all_rows[i].find_all("td") :
            table_data[i][j] = k.text
            j += 1

    # print(table_data)
        

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


def parse_table(document) :
    if not document or document == "{}" or document["table"] == "{}":
        table_whole = []

    html_data = document["table"].replace('\"', '').replace("\n", " ")
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

        table_head = extract_head(table_head, row_len)

        table_whole = [table_head, *table_body]

    except:
        table_whole = []

    return table_whole


age_similarity_table = {}

biobert = BioBert(model_path = "../num_data")

def individual_embedding(text):
    vectors, tokens = biobert.word_vector(text)
    b = vectors[0]
    for m in vectors[1: ] :
        b += m
    b /= len(tokens)
    b = [k.item() for k in b]

    return np.array(b)

def compact_embeddings(att, num, unit):
    att_emb = individual_embedding(str(att).lower())
    num_emb = individual_embedding(str(num).lower())
    unit_emb = individual_embedding(str(num).lower())

    emb = [*att_emb, *num_emb, *unit_emb]

    return np.array(emb)

age_similarity_table = pd.DataFrame(columns = ["paper_ID", "table_ID", "Attribute", "similarity"])

age_ref = [
    ["Mean age (years)", 27.8, "years"],
    ["Mean age (years)", 21.3, "years"],
    ["Mean age (years)", 22.4, "years"],
    ["Mean age (years)", 23.8, "years"],
    ["Mean age (years)", 24.73, "years"],
]

age_emb = 0
for i in age_ref :
    if isinstance(age_emb, int) :
        age_emb = compact_embeddings(*i)
    else :
        age_emb += compact_embeddings(*i)
age_emb /= len(age_ref)

mortality_ref = [
    ["mortality rates : Crude : 2009", 8.0, "NO UNITS"],
    ["mortality rates : Crude : 2009", 10.1, "NO UNITS"],
    ["mortality rates : Crude : 2009", 9.7, "NO UNITS"],
    ["mortality rates : Crude : 2009", 9.9, "NO UNITS"],
    ["mortality rates : Crude : 2009", 11.8, "NO UNITS"],
    ["mortality rates : Crude : 2009", 14.9, "NO UNITS"],
    ["mortality rates : Crude : 2009", 10.9, "NO UNITS"],
    ["mortality rates : Crude : 2009", 10.6, "NO UNITS"],
    ["mortality rates : Crude : 2009", 11.1, "NO UNITS"],
    ["mortality rates : Crude : 2009", 12.0, "NO UNITS"],
    ["mortality rates : Crude : 2009", 10.9, "NO UNITS"],
    ["mortality rates : Crude : 2009", 10.3, "NO UNITS"],
    ["mortality rates : Crude : 2009", 12.3, "NO UNITS"],
    ["mortality rates : Crude : 2009", 9.5, "NO UNITS"],
    ["mortality rates : Crude : 2009", 13.1, "NO UNITS"],
    ["mortality rates : Crude : 2009", 13.1, "NO UNITS"]
]

mortality_emb = 0
for i in mortality_ref :
    if isinstance(mortality_emb, int) :
        mortality_emb = compact_embeddings(*i)
    else :
        mortality_emb += compact_embeddings(*i)
mortality_emb /= len(mortality_emb)


positives = [
    {
        "$unwind":"$tables"
    },
    {
    "$project": {
      "_id": 1,
      "title": 1,
      "table_id": "$tables.tableId",
      "table": "$tables.tableData"
                }
    }
]

count = 0

positives_cursor = colorectal_kg.aggregate(positives)
for document in positives_cursor:
    # print(document)

    count += 1

    table = parse_table(document)
    
    if table:

        for i in range(len(table[0])) :

            if table[0][i].strip() == "" :
                continue

            unit_1 = return_units(table[0][i])

            emb = 0

            for j in range(1, len(table)) :
                if table[j][i] == 0:
                    continue
                
                table[j][i] = str(table[j][i])

                if table[j][i].lower().strip() in ["-", "\u2013", "na", "n/a", "nan", ]:
                    table[j][i] = "null"
                
                if table[j][i].strip() == "" :
                    continue
                
                unit_2 = return_units(table[j][i])

                unit = unit_2 if unit_2 else (unit_1 if unit_1 else "UNKNOWN")
                embedding = individual_embedding(table[j][i])

                if isinstance(emb, int) :
                    emb = embedding
                else :
                    emb += embedding
                
                emb /= (len(table) - 1)

                attr_emb = individual_embedding(table[0][i])
                unit_emb = individual_embedding(unit)

            # print(cosine_similarity(emb.reshape(1, -1), age_emb.reshape(1, -1)))

            if not isinstance(emb, int) : 

                emb = np.concatenate([attr_emb, emb, unit_emb])

                age_similarity_table = age_similarity_table.append({"paper_ID": document["_id"], "table_ID": document["table_id"], "Attribute": table[0][i], "age_similarity": abs(round(1 - cosine(np.array(emb), np.array(age_emb)), 6)), "mortality_similarity": abs(round(1 - cosine(np.array(emb), np.array(mortality_emb)), 6))}, ignore_index = True)
    
    if count % 1000 == 0 : 

        print(count)

        age_similarity_table2 = age_similarity_table[["paper_ID", "table_ID", "Attribute", "age_similarity"]].sort_values(by='age_similarity', ascending=False)

        age_similarity_table2.to_csv("age_similarity.csv", ignore_index=True)

        age_similarity_table = age_similarity_table[["paper_ID", "table_ID", "Attribute", "mortality_similarity"]].sort_values(by='mortality_similarity', ascending=False)

        age_similarity_table.to_csv("mortality_similarity.csv", ignore_index = True)

print("Com[ple]ted")