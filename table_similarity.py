import pandas as pd
import numpy as np

colorectal = pd.read_csv("colorectalKG.csv", delimiter="~", names=["id", "table_id", "data"])

from biobert_embeddings import BioBert
from scipy.spatial.distance import cosine

biobert = BioBert("../num_data")

def embedding(data) :
    
    emb, tok = biobert.word_vector(data)
    b = emb[0]

    for m in emb[1: ] :
        b += m

    b /= len(tok)

    return np.array([k.item() for k in b])

ref1 = colorectal[(colorectal["id"] == "6467a7740ce57759d3714e69") & (colorectal["table_id"] == 1)]

ref2 = colorectal[(colorectal["id"] == "6467a0a70ce57759d3660e4c") & (colorectal["table_id"] == 2)]

ref3 = colorectal[(colorectal["id"] == "6467ac980ce57759d37aaf28") & (colorectal["table_id"] == 1)]

ref1_emb = embedding(str(ref1["data"]))
ref2_emb = embedding(str(ref2["data"]))
ref3_emb = embedding(str(ref3["data"]))

ref1_similarities = []
ref2_similarities = []
ref3_similarities = []

count = 0
for i in colorectal["data"]:
    # print(i)
    try:
        curr_emb = embedding(i.strip())

        ref1_similarities.append(abs(round(1 - cosine(curr_emb, ref1_emb), 6)))
        ref2_similarities.append(abs(round(1 - cosine(curr_emb, ref2_emb), 6)))
        ref3_similarities.append(abs(round(1 - cosine(curr_emb, ref3_emb), 6)))
    
    except Exception as e:
        count += 1
        print(e, count)
        print(i)
        ref1_similarities.append(0)
        ref2_similarities.append(0)
        ref3_similarities.append(0)
        

colorectal = colorectal[["id", "table_id"]]
colorectal["ref1_similarity"] = ref1_similarities
colorectal["ref2_similarity"] = ref2_similarities
colorectal["ref3_similarity"] = ref3_similarities

colorectal.to_csv("table_similarities.csv")
print(count, "tables are not included")