import pandas as pd
import numpy as np

from scipy.spatial.distance import cosine

words_to_search = ["cancer", "paclitaxel", "Sorafenib", "Cisplatin", "Niraparib", "Colorectal", "Carcinoma", "Sarcoma", "Melanoma", "fever", "cough"]

data_columns = [('a' + str(i)) for i in range(1, 769)]

def cosine_similarity(a, b) :

    return abs(round(1 - cosine(np.array(a), np.array(b)), 6))

def find_vectors(data, word):
    if not isinstance(word, str):
        print("Give a word to search not some ....")

    req = data[data["word"] == word]

    if req.shape[0] == 0 :
        print("No such word exixts in the dataset")

    return req

def preprocess(data):

    data.dropna(inplace=True)

    for i in data_columns :
        data[i] = data[i].astype(float)

    return data

print("Reading Files....")

print("Reading meta_data.....")
meta_data = pd.read_csv("./embeddings_clusters/cancerkg_clusters_meta_v_md.tsv", sep  = '\t')

print("Reading data.........")
data = pd.read_csv("./embeddings_clusters/cancerkg_clusters_meta_v.tsv", sep = '\t', names = data_columns)


print("Preprocessing data.......")
data = preprocess(data)


print("Making vectors for data")
meta_data["vectors"] = data[data_columns].apply(list, axis=1)
data = meta_data

for word_to_search in words_to_search :
    print("Finding vectors related to the word:", word_to_search)
    required_vectors = find_vectors(data, word_to_search)
    print("search_results_of_word: ", word_to_search, ":", required_vectors.shape[0])

    print("Computing Similar Vectors")
    print("**********************")
    for j in range(required_vectors.shape[0]) :
        data["similarity"] = data["vectors"].apply(lambda x : cosine_similarity(x, required_vectors.iloc[j, 2]))

        top_10_nearest_values = data.nlargest(10, "similarity")["similarity"].tolist()

        top_10_nearest_values.sort(reverse = True)

        print("######", top_10_nearest_values, "#######")
        # print(data)

        similar_words = data[data.apply(lambda x: x["similarity"] in (top_10_nearest_values), axis=1)][["word", "text", "similarity"]]

        print("************************")
        print(required_vectors.iloc[j, :2])
        print(similar_words)
        print("************************")
        print("************************")

    print("|||||||||||||||||||||||||||||||||||")
    print("###################################")
    print("|||||||||||||||||||||||||||||||||||")

print("Com[ple]ted")