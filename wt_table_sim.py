import pandas as pd
import numpy as np

import pickle

colorectal = pd.read_csv("web_tables.toksv", delimiter="~")
colorectal.dropna(inplace=True)

print(colorectal.head())

from biobert_embeddings import BioBert
from scipy.spatial.distance import cosine

biobert = BioBert("../models/web_tables")

def embedding(data) :
    data = str(data)

    # print(data)

    data = eval(data)
    b = 0

    for i in range(len(data)):
        x = " ".join(data[i])

        emb, tok = biobert.word_vector(x)
        b = emb[0]

        for m in emb[1: ] :
            b += m

    b /= len(tok)

    return np.array([k.item() for k in b])

ref1 = colorectal[colorectal["id"] == "60a52fd7bf30c995ef184e6c"]

ref2 = colorectal[colorectal["id"] == "60a52fcfbf30c995ef1841b3"]

ref3 = colorectal[colorectal["id"] == "60a52fbbbf30c995ef18206c"]

ref4 = colorectal[colorectal["id"] == "60a52fd5bf30c995ef184b9c"]

ref5 = colorectal[colorectal["id"] == "60a52fc8bf30c995ef1834da"]

ref1_emb = embedding(ref1.iloc[0][1])
ref2_emb = embedding(ref2.iloc[0][1])
ref3_emb = embedding(ref3.iloc[0][1])
ref4_emb = embedding(ref4.iloc[0][1])
ref5_emb = embedding(ref5.iloc[0][1])

ref1_similarities = []
ref2_similarities = []
ref3_similarities = []
ref4_similarities = []
ref5_similarities = []

count = 0
for ii in colorectal["table"]:

    count += 1

    try:
        curr_emb = embedding(ii)

        a = abs(round(1 - cosine(curr_emb, ref1_emb), 6))
        b = abs(round(1 - cosine(curr_emb, ref2_emb), 6))
        c = abs(round(1 - cosine(curr_emb, ref3_emb), 6))
        d = abs(round(1 - cosine(curr_emb, ref4_emb), 6))
        e = abs(round(1 - cosine(curr_emb, ref5_emb), 6))

        print(ii, a, b, c, d, e)



    except Exception as e:
        print(e, count)
        # exit(0)
        # print(i)
        # ref1_similarities.append(0)
        # ref2_similarities.append(0)
        # ref3_similarities.append(0)
        # ref4_similarities.append(0)
        # ref5_similarities.append(0)


    if count % 100 == 0:
        print(count)

#         pickle.dump(ref1_similarities, open("wt_ref1_similarities", "wb"))
#         pickle.dump(ref2_similarities, open("wt_ref2_similarities", "wb"))
#         pickle.dump(ref3_similarities, open("wt_ref3_similarities", "wb"))
#         pickle.dump(ref4_similarities, open("wt_ref4_similarities", "wb"))
#         pickle.dump(ref5_similarities, open("wt_ref5_similarities", "wb"))

# pickle.dump(ref1_similarities, open("wt_ref1_similarities", "wb"))
# pickle.dump(ref2_similarities, open("wt_ref2_similarities", "wb"))
# pickle.dump(ref3_similarities, open("wt_ref3_similarities", "wb"))
# pickle.dump(ref4_similarities, open("wt_ref4_similarities", "wb"))
# pickle.dump(ref5_similarities, open("wt_ref5_similarities", "wb"))


print(count, "tables are not included")