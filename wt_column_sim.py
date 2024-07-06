import pandas as pd

import sys
import re
import json
import os
import torch
import numpy as np
import pickle

from pymongo import MongoClient
from bs4 import BeautifulSoup
from scipy.spatial.distance import cosine
from transformers import BertModel, BertTokenizer

from biobert_embeddings import BioBert # User Defined Class

biobert = BioBert(model_path = "../models/web_tables")

def individual_embedding(text):
    vectors, tokens = biobert.word_vector(text)
    b = vectors[0]
    for m in vectors[1: ] :
        b += m
    b /= len(tokens)
    b = [k.item() for k in b]

    return np.array(b)

similarity_table = pd.DataFrame(columns = ["ID", "table",])

ref_column_1 = [
    'Grant or Scholarship Aid', '$5,593', '79%'
]

ref_emb_1 = 0
for i in ref_column_1 :
    if isinstance(ref_emb_1, int) :
        # ref_emb = compact_embeddings(*i)
        ref_emb_1 = np.array(individual_embedding(i))
    else :
        # ref_emb += compact_embeddings(*i)
        ref_emb_1 += np.array(individual_embedding(i))

# ref_emb = np.array(ref_emb)
ref_emb_1 /= len(ref_column_1)

ref_column_2 = [
    'No strong or coarse Language', 
    'Infrequent/Mild profanity or crude humor',
    'Intense or graphic profanity/crude humor',
    'BlackBerry World will not accept Hate Speech'
]

ref_emb_2 = 0
for i in ref_column_2 :
    if isinstance(ref_emb_2, int) :
        # ref_emb = compact_embeddings(*i)
        ref_emb_2 = np.array(individual_embedding(i))
    else :
        # ref_emb += compact_embeddings(*i)
        ref_emb_2 += np.array(indiavidual_embedding(i))

# ref_emb = np.array(ref_emb)
ref_emb_2 /= len(ref_column_2)

ref_column_3 = [
    'Company', 'Country', 'Industry', 'sales ($bil)',
    'Profits ($bil)', 'Assets ($bil)', 'Market Value ($bil)',
]

ref_emb_3 = 0
for i in ref_column_3 :
    if isinstance(ref_emb_3, int) :
        # ref_emb = compact_embeddings(*i)
        ref_emb_3 = np.array(individual_embedding(i))
    else :
        # ref_emb += compact_embeddings(*i)
        ref_emb_3 += np.array(individual_embedding(i))

# ref_emb = np.array(ref_emb)
ref_emb_3 /= len(ref_column_3)

df = pd.read_csv("web_tables.toksv", sep="~")

for ii in range(df.shape[0]) :
    record = df.iloc[ii][:]
    table_id = record.iloc[0]
    table = record.iloc[1]

    table = eval(table)

    for i in range(len(table[0])) :

        emb = 0

        for j in range(1, len(table)) :

            table[j][i] = str(table[j][i])

            if table[j][i].strip() == "" :
                continue

            try:
                embedding = individual_embedding(table[j][i])

            except:
                embedding = 0
                # break

            try:
                if isinstance(emb, int) :
                    emb = np.array(embedding, dtype=float)
                else :
                    emb += np.array(embedding, dtype=float)

            except:
                continue

            emb /= (len(table) - 1)

            #print(round(1 - cosine(np.array(emb), np.array(ref_emb)), 6))

        if not isinstance(emb, int) :
            try:
                ref_1_sim = abs(round(1 - cosine(np.array(emb), np.array(ref_emb_1)), 6))
                ref_2_sim = abs(round(1 - cosine(np.array(emb), np.array(ref_emb_2)), 6))
                ref_3_sim = abs(round(1 - cosine(np.array(emb), np.array(ref_emb_3)), 6))

                similarity_table = similarity_table.append({"ID": table_id, "table": table, "Attribute": table[0][i], "ref_1_similarity": abs(round(1 - cosine(np.array(emb), np.array(ref_emb_1)), 6)), "ref_2_similarity": abs(round(1 - cosine(np.array(emb), np.array(ref_emb_2)), 6)), "ref_3_similarity": abs(round(1 - cosine(np.array(emb), np.array(ref_emb_3)), 6))}, ignore_index = True)

            except:
                pass

    if ii % 200 == 0 :
      print(ii)
      similarity_table.to_pickle("wt_column_sim_pickle.pkl")

print("count What h appended?")
similarity_table.to_csv("wt_column_similarity.csv")