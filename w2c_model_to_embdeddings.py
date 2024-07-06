import gensim
import csv

import numpy as numpy

model = gensim.models.Word2Vec.load("path")

data_ = open("word2_vec_data.tsv", 'w+')
datawriter = csv.writer(data_, sep = '\t')

meta = open("word2_vec_data_md.txt", 'w+')

for index, word in enumerate(model.wv.index_to_key):
    if index == 10:
        break
    meta.write(word.strip() + "\n")
    datawriter.writerow(np.array(model.wv[word]))