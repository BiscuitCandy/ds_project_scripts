import pandas as pd
import numpy as np

colorectal = pd.read_csv("ColorectalKG2.csv", delimiter="~", names=["id", "table_id", "data"])

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

ref4 = colorectal[(colorectal["id"] == "64679db40ce57759d3620cc8") & (colorectal["table_id"] == 1)]

ref5 = colorectal[(colorectal["id"] == "6467ac980ce57759d37aaf28") & (colorectal["table_id"] == 5)]

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
for ii in colorectal["data"]:
    i = eval(ii)
    # print(i)
    curr_emb = 0
    try:
        for j in range(len(i)):
            # print(i[j])
            if not i[j].strip():
                continue
            if isinstance(curr_emb, int) :
                curr_emb = embedding(i[j].strip())
            else :
                curr_emb += embedding(i[j].strip())

        curr_emb /= len(i)


        ref1_similarities.append(abs(round(1 - cosine(curr_emb, ref1_emb), 6)))
        ref2_similarities.append(abs(round(1 - cosine(curr_emb, ref2_emb), 6)))
        ref3_similarities.append(abs(round(1 - cosine(curr_emb, ref3_emb), 6)))
        ref4_similarities.append(abs(round(1 - cosine(curr_emb, ref4_emb), 6)))
        ref5_similarities.append(abs(round(1 - cosine(curr_emb, ref5_emb), 6)))

    except Exception as e:
        count += 1
        # print(e, count)
        # exit(0)
        # print(i)
        ref1_similarities.append(0)
        ref2_similarities.append(0)
        ref3_similarities.append(0)
        ref4_similarities.append(0)
        ref5_similarities.append(0)


colorectal = colorectal[["id", "table_id"]]
colorectal["ref1_similarity"] = pd.Series(ref1_similarities)
colorectal["ref2_similarity"] = pd.Series(ref2_similarities)
colorectal["ref3_similarity"] = pd.Series(ref3_similarities)
colorectal["ref4_similarity"] = pd.Series(ref4_similarities)
colorectal["ref5_similarity"] = pd.Series(ref5_similarities)

colorectal.to_csv("table_similarities2.csv")
print(count, "tables are not included")