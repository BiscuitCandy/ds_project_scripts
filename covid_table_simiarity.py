import pandas as pd
import numpy as np

import sys

import pickle

from bs4 import BeautifulSoup

colorectal = pd.read_csv("covid_tables.toksv", delimiter="~")
# print(colorectal)

from biobert_embeddings import BioBert
from scipy.spatial.distance import cosine

biobert = BioBert("../models/biobert_CKG/")

def extract_body(table_body) :

    all_rows = table_body.find_all("tr")
    total_rows_len = len(all_rows)

    first_row = table_body.find("tr").find_all("td")

    row_len = 0
    for i in first_row :
        row_len += int(str(i["colspan"]).replace('"', '').replace("'", "").replace("/", "").replace("\\", "")) if i.get("colspan") else 1

    table_data = [[0 for i in range(row_len)] for j in range(total_rows_len)]

    for i in range(len(all_rows)) :

        j = 0

        for k in all_rows[i].find_all("td") :
            table_data[i][j] = str(k.text)
            j += 1

    # print(table_data)


    return table_data


def extract_head(table_head, row_len):

    head_rows = table_head.find_all("tr")
    head_rows_len = len(head_rows)

    table_head = [[0 for i in range(row_len)] for j in range(head_rows_len)]
    # print(table_head)

    # print("table_head-start")

    for i in range(len(head_rows)) :

        j = 0

        for k in head_rows[i].find_all("td"):
            table_head[i][j] = str(k.text)
            j += 1

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

    # print(head)

    return head


def parse_table(document) :
    if not document or document == "{}" or document == "nan":
        table_whole = []
        return table_whole

    html_data = document.replace('\"', '').replace("\n", " ")
    html_data = html_data.replace("<th ", "<td ")
    html_data = html_data.replace("<th>", "<td>")
    html_data = html_data.replace("</th>", "</td>")

    # print(html_data)

    soup = BeautifulSoup(html_data, 'html.parser')

    table_body = soup.tbody
    table_head = soup.thead

    try:
        table_body = extract_body(table_body)
        # except : table_body = []
        row_len = len(table_body[0])
        # print(table_body)

        try: table_head = extract_head(table_head, row_len)
        except: table_head = []
        # print(table_head if table_head else "***********************")

        table_whole = [table_head, *table_body] if table_head else table_body

    except Exception as e:
        print(e, document)
        table_whole = []

    return table_whole

def embedding(data) :

    emb, tok = biobert.word_vector(data)
    b = emb[0]

    for m in emb[1: ] :
        b += m

    b /= len(tok)

    return np.array([k.item() for k in b])

ref1 = colorectal[(colorectal["id"] == "64b44f1a6e6abfba38b4f95f") & (colorectal["table"] == 1)]

ref2 = colorectal[(colorectal["id"] == "64b44f196e6abfba38b4f80f") & (colorectal["table"] == 3)]

ref3 = colorectal[(colorectal["id"] == "64b44f156e6abfba38b4e96a") & (colorectal["table"] == 7)]

ref4 = colorectal[(colorectal["id"] == "64b44e996e6abfba38b36e69") & (colorectal["table"] == 1)]

ref5 = colorectal[(colorectal["id"] == "64b44f196e6abfba38b4f8aa") & (colorectal["table"] == 3)]

ref1_emb = embedding(str(ref1["data"]))
ref2_emb = embedding(str(ref2["data"]))
ref3_emb = embedding(str(ref3["data"]))
ref4_emb = embedding(str(ref4["data"]))
ref5_emb = embedding(str(ref5["data"]))

ref1_similarities = []
ref2_similarities = []
ref3_similarities = []
ref4_similarities = []
ref5_similarities = []

count = 0
for i in colorectal["data"]:
    count += 1
    try: 
        # print(str(i))
        table_i = parse_table(str(i))
        curr_emb = 0
        # print(table_i)
        for j in table_i :
            row_j = " ".join([str(k) for k in j if j != 0])
            # print(row_j)
            if isinstance(curr_emb, int):
                curr_emb = embedding(str(row_j).strip())
            else :
                curr_emb += embedding(str(row_j).strip())

            
        if not isinstance(curr_emb, int):
            ref1_similarities.append(abs(round(1 - cosine(curr_emb, ref1_emb), 6)))
            ref2_similarities.append(abs(round(1 - cosine(curr_emb, ref2_emb), 6)))
            ref3_similarities.append(abs(round(1 - cosine(curr_emb, ref3_emb), 6)))
            ref4_similarities.append(abs(round(1 - cosine(curr_emb, ref4_emb), 6)))
            ref5_similarities.append(abs(round(1 - cosine(curr_emb, ref5_emb), 6)))

    except Exception as e:
        print(e, count)
        # print(i)
        ref1_similarities.append(0)
        ref2_similarities.append(0)
        ref3_similarities.append(0)
        ref4_similarities.append(0)
        ref5_similarities.append(0)

    if count % 1000 == 0:
        # colorectal = colorectal[["id", "table", "table_caption"]]
        # colorectal["ref1_similarity"] = pd.Series(ref1_similarities)
        # colorectal["ref2_similarity"] = pd.Series(ref2_similarities)
        # colorectal["ref3_similarity"] = pd.Series(ref3_similarities)
        # colorectal["ref4_similarity"] = pd.Series(ref4_similarities)
        # colorectal["ref5_similarity"] = pd.Series(ref5_similarities)

        pickle.dump(ref1_similarities, open("ref1_similarities", "wb"))
        pickle.dump(ref2_similarities, open("ref2_similarities", "wb"))
        pickle.dump(ref3_similarities, open("ref3_similarities", "wb"))
        pickle.dump(ref4_similarities, open("ref4_similarities", "wb"))
        pickle.dump(ref5_similarities, open("ref5_similarities", "wb"))

pickle.dump(ref1_similarities, open("ref1_similarities", "wb"))
pickle.dump(ref2_similarities, open("ref2_similarities", "wb"))
pickle.dump(ref3_similarities, open("ref3_similarities", "wb"))
pickle.dump(ref4_similarities, open("ref4_similarities", "wb"))
pickle.dump(ref5_similarities, open("ref5_similarities", "wb"))

d ={}
d["ref1"] = ref1_similarities
d["ref2"] = ref2_similarities
d["ref3"] = ref3_similarities
d["ref4"] = ref4_similarities
d["ref5"] = ref5_similarities

df = pd.DataFrame(d)

df.to_csv("covid_tables_similarity.csv")
print("Com[ple]ted")