from pymongo import MongoClient
import csv
import sys
import re
import json
import hashlib
import pickle

from bs4 import BeautifulSoup

ATTRIBUTE_UNIT_LIST = {}


client = MongoClient('mongodb://readUser:mongo232@bl4:27017/?authSource=test&authMechanism=SCRAM-SHA-256')

db = client['test']

colorectal_kg = db['Colorectal_KG']

range_pattern = r'\b(\d+(\.\d+)?)[-\u2013](\d+(\.\d+)?)\b'
range_pattern = re.compile(range_pattern)

number_pattern = r'(-?(\d+(\.\d+)?))'
number_pattern = re.compile(number_pattern)

unit_pattern = {}

units = ['years', 'days', 'min', 'ml/h/kg', 'mg*h/l', 'hb/g', 'ng/ml', 'ug/ml', 'pg/ml', 'mg/ml', 'mg/kg', 'ug/l', 'kg/m2', 'kg/l', 'kg/m', 'kg/m3', 'kg/m2', 'g/m' 'pg/l', 'ng/l', 'g/l', 'g/m', 'ml', 'um', 'ug', 'nm', 'mm','cm', 'km', 'kg', 'pg', 'mg', 'Gb']

for i in units :
    pattern = '[A-Za-z0-9]*\s+' + i +'\s+[A-Za-z0-9]*'

    p = re.compile(pattern, re.IGNORECASE)

    unit_pattern[p] = i

def return_units(data):
    for i in unit_pattern :
        if i.search(str(data)) : 
            return unit_pattern[i]

def check_range(data):
    return range_pattern.findall(str(data))

def check_number(data):
    return number_pattern.findall(str(data))


def generate_unique_token(attribute, unit, r1, r2):
    
    input_string = f"{attribute}|{unit}|{r1}|{r2}"

    
    sha256_hash = hashlib.sha256(input_string.encode()).digest()

    
    custom_charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    
    token = ''.join([custom_charset[i % len(custom_charset)] for i in sha256_hash])

    return token

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
            value = str(table_head[j][i]).strip()
            if char == "":
                char = char + value

            else :
                if value == char :
                    pass

                else :
                    char = char + " : " + value

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

        # print(table_whole)
        n += 1
    except:
        table_whole = []
        # print("A table can not be represented", n+1)
        # print("*"*20)

    if table_whole :
        for i in range(len(table_whole[0])) :
            unit_1 = return_units(table_whole[0][i])
            # print(table_whole[0][i])

            for j in range(1, len(table_whole)):
                num_range = check_range(table_whole[j][i])
                unit_2 = return_units(table_whole[j][i])
                if num_range and (unit_1 or unit_2) :
                    print(num_range, "|||", unit_1, " -- ",  unit_2, "|||", table_whole[0][i], "|||", table_whole[j][i])

                    unit = unit_1 if unit_1 else unit_2

                    for num_range_i in num_range:

                        table_whole[j][i] = re.sub(num_range_i[0] + "-" + num_range_i[2], " " + generate_unique_token(table_whole[0][i], unit, float(num_range_i[0]), float(num_range_i[2])) + " ", table_whole[j][i])
                        table_whole[j][i] = re.sub(num_range_i[0] + " " + num_range_i[2], " " + generate_unique_token(table_whole[0][i], unit, float(num_range_i[0]), float(num_range_i[2])) + " ", table_whole[j][i])

                        # print(table_whole[j][i])

                        if (table_whole[0][i], unit) in ATTRIBUTE_UNIT_LIST:
                            if ((num_range_i[0], num_range_i[2])) in ATTRIBUTE_UNIT_LIST[(table_whole[0][i], unit)] :
                                continue
                            else :
                                ATTRIBUTE_UNIT_LIST[(table_whole[0][i], unit)].append((num_range_i[0], num_range_i[2]))
                        else :
                            ATTRIBUTE_UNIT_LIST[(table_whole[0][i], unit)] = [(num_range_i[0], num_range_i[2]), ]
            
        for i in range(len(table_whole[0])) :
            unit_1 = return_units(table_whole[0][i])
            # print(table_whole[0][i])

            for j in range(1, len(table_whole)):
                unit_2 = return_units(table_whole[j][i])
                num_number = check_number(table_whole[j][i])
                num_range = check_range(table_whole[j][i])

                if num_number and (unit_1 or unit_2) and not num_range:
                    unit = unit_1 if unit_1 else unit_2
                    if (table_whole[0][i], unit) in ATTRIBUTE_UNIT_LIST :
                        
                        for k in ATTRIBUTE_UNIT_LIST[(table_whole[0][i], unit)] :
                            for num_number_i in num_number:
                                if float(k[0]) <= float(num_number_i[0]) <= float(k[1]) :
                                    table_whole[j][i] = re.sub("\\b" + num_number_i[0] + "\\b", " " + generate_unique_token(table_whole[0][i], unit, float(k[0]), float(k[1])) + " ", table_whole[j][i])
                                    print("TTTTTTT", table_whole[0][i], "|||",  unit, "|||",  num_number_i, "---", table_whole[j][i])

        for i in range(len(table_whole)) :
                if i == 0:
                    hdm.write(" ".join(table_whole[0]) + "\n")
                    

with open('attriute_unit_list.pkl', 'wb') as file:
    pickle.dump(ATTRIBUTE_UNIT_LIST, file)

