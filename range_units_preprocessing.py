from pymongo import MongoClient
import csv
import sys
import re
import json

# connect to MongoDB and target cord19_%
client = MongoClient('mongodb://readUser:mongo232@bl4:27017/?authSource=test&authMechanism=SCRAM-SHA-256')

db = client['test']
# select the database
colorectal_kg = db['Colorectal_KG']

number_pattern = r'[a-zA-Z]+\d+[a-zA-Z\d]*|[a-zA-Z]*\d+[a-zA-Z]+'
number_pattern = re.compile(number_pattern)

range_pattern = r'\b(\d+(\.\d+)?)[-\u2013](\d+(\.\d+)?)\b'
range_pattern = re.compile(range_pattern)

unknown_file = open("tex_num_tex2_units.format", 'w+')

table_data = []
unq_tkn = 0
print("unq_tkn_"+str(unq_tkn))

unique_attributes = {}
unit_pattern = {}

units = ['years', 'days', 'ml/h/kg', 'mg*h/l', 'hb/g', 'ng/ml', 'ug/ml', 'pg/ml', 'mg/ml', 'mg/kg', 'ug/l', 'kg/m2', 'kg/l', 'kg/m3', 'kg/m2', 'kg/m', 'mg/m3', 'mg/m2', 'mg/m' 'g/m3', 'g/m2', 'g/m' 'pg/l', 'ng/l', 'g/l', 'g l-1', 'g/m', 'ml', 'cm2', 'cm', 'km', 'kg', 'pg', 'mg', 'g', 'l', 'm3', 'm2', 'm']

for i in units :
    pattern = '[A-Za-z0-9]*\s+' + i +'\s+[A-Za-z0-9]*'

    p = re.compile(pattern, re.IGNORECASE)

    unit_pattern[p] = i

def check_range(data):

    # print(data)
    # print(range_pattern.search(data))
    x = range_pattern.findall(data)

    return x

def check_text_num_text_format(data) :

    try: 
        x = number_pattern.findall(data)
    # print("texnumtex", x)

    except:
        x = []

    for i in x :
        unknown_file.write(i + "\n")

def return_units(data):
    
    try:
        for i in unit_pattern :
            if i.search(data) : 
                print(data, unit_pattern[i])
                return unit_pattern[i]
    except: pass    
    return None


def add_units(data, unit):
    
    return (data + " " + unit)

def createDataset(document, idx, positive_condition):
    global unq_tkn
    row_data = {}
    row_data["caption"] = document["caption"]
    # row_data["cl_HMD"] = document["cl_HMD"] if "cl_HMD" in document else []
    # row_data["cl_VMD"] = document["cl_VMD"] if "cl_VMD" in document else []
    # row_data["cl_DATA"] = document["cl_DATA"] if "cl_DATA" in document else []

    if "data" in document and "cl_HMD" in document:

        # for i in ["cl_HMD", "cl_VMD", "data"]: 
        #     document[i] = document[i][0]

        hmd = document["cl_HMD"]
        real_hmd = [j for i in  hmd if i for j in i] if hmd else []
        data = document["data"] if "data" in document else []

        if data == [] :
            row_data["data"] = data
            row_data["meta_v"] = document["cl_VMD"]
            row_data["meta_h"] = real_hmd

            table_data.append(row_data)

            return idx

        # now add the units for the values based on attribute units
        range_label = 0
        unit = None

        try:
            for i in range(len(data[0])) :
                try:
                    unit = return_units(str(real_hmd[i]))
                    check_text_num_text_format(str(real_hmd[i]))
                except:
                    pass
                for j in range(len(data)) :
                    if unit:
                        try: data[j][i] = add_units(str(data[j][i]), unit)
                        except: continue

                    ## also check if data[j][i] is a range or not
                    ## give range_label = 1 if the value is range else 0

                    try: 
                        range_label = check_range(str(data[j][i]))[0]
                        if range_label: 
                            key = real_hmd[i] if (i < len(real_hmd) and real_hmd[i] != "nan") else "unidentified"
                            token = ("unq_tkn_"+str(unq_tkn))
                            # print(range_label, key, range_label[0], range_label[1])
                            unit = return_units(data[j][i]) if not unit else unit
                            value = [range_label[0], range_label[2], (unit if unit else "unknown")]
                            if key in unique_attributes:
                                flag = 0
                                for i in unique_attributes[key]:
                                    if i[:3] == value[:3]:
                                        flag = 1
                                        break
                                if flag == 0:
                                    # print(key, value)
                                    value.append(token)
                                    unique_attributes[key].append(value)
                                    unq_tkn += 1
                            else :
                                unique_attributes[key] = [value, ]

                    except: continue

                    ## also check if data[j][i] is having numericals included in text
                    ## add these terms to a seperate file

                    try: check_text_num_text_format(str(data[j][i]))
                    except: pass
        except: pass

        ## finally after modifing all these
        ## now we write this to the csv file
        row_data["data"] = data
        row_data["meta_v"] = document["cl_VMD"]
        row_data["meta_h"] = real_hmd

    table_data.append(row_data)

    idx += 1
    return idx

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
      "caption": "$tables.tableCaption",
      "cl_VMD": "$tables.cl_VMD",
      "cl_HMD": "$tables.cl_HMD",
      "data": "$tables.cl_DATA"
                }
    }
]

positives_cursor = colorectal_kg.aggregate(positives)

row_count=0
with open('./test_table_format2_units', 'w', encoding='utf-8') as outfile:
  for document in positives_cursor:
    # print(document)
    row_count = createDataset(document, row_count, positive_condition= True)

  outfile.write(json.dumps(table_data))

with open("range_attributes_units.tokens", "w") as f :
    f.write(json.dumps(unique_attributes))

print(f'Total rows in positive set:\t{row_count}')