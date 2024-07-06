import pandas as pd

data = pd.DataFrame(columns = ["word", "text", "vectors"])

from scipy.spatial.distance import cosine
import os
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
import sys

def cosine_similarity(a, b) :

    return abs(round(1 - cosine(np.array(a), np.array(b)), 6))

def find_vectors(data, word):
    if not isinstance(word, str):
        print("Give a word to search not some ....")

    req = data[data["word"].str.contains(word)]
    # req = req[req["word"].str.contains(unit)]

    if req.shape[0] == 0 :
        print("No such word exixts in the dataset")

    return req

class BioBert(object):
    def __init__(self, model_path):
        if model_path is not None:
            self.model_path = model_path
        else:
            print("specify one of the following [DATA, META_V, META_H] as model based on the input data")

        self.tokens = ""
        self.sentence_tokens = ""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertModel.from_pretrained(self.model_path, output_hidden_states = True)

    def process_text(self, text):

        marked_text = "[CLS] " + text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        return tokenized_text


    def handle_oov(self, tokenized_text, word_embeddings):
        embeddings = []
        tokens = []
        oov_len = 1
        for token,word_embedding in zip(tokenized_text, word_embeddings):
            if token.startswith('##'):
                token = token[2:]
                tokens[-1] += token
                oov_len += 1
                embeddings[-1] += word_embedding
            else:
                if oov_len > 1:
                    embeddings[-1] /= oov_len
                tokens.append(token)
                embeddings.append(word_embedding)
        return tokens,embeddings


    def eval_fwdprop_biobert(self, tokenized_text):

        segments_ids = [1] * len(tokenized_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        self.model.eval()

        with torch.no_grad():
            encoded_layers = self.model(tokens_tensor, segments_tensors)

        return encoded_layers[2]


    def word_vector(self, text, handle_oov=True, filter_extra_tokens=True):

        tokenized_text = self.process_text(text)

        encoded_layers = self.eval_fwdprop_biobert(tokenized_text)

        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)

        word_embeddings = []
        for token in token_embeddings:
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            word_embeddings.append(sum_vec)

        self.tokens = tokenized_text
        if filter_extra_tokens:
            # filter_spec_tokens: filter [CLS], [SEP] tokens.
            word_embeddings = word_embeddings[1:-1]
            self.tokens = tokenized_text[1:-1]

        if handle_oov:
            self.tokens, word_embeddings = self.handle_oov(self.tokens,word_embeddings)

        return ((word_embeddings), self.tokens)


import pickle

d = pickle.load(open("cancerkg_attributes_list.pkl", 'rb'))

att_unit = list(d.keys())

biobert = BioBert(model_path = "./models/cancerkg_new_biobert_seq_64/data/")

for i in att_unit :

    if len(i[0]) > 500 :
        continue

    att, unit = i

    if att.strip() in ["0", "", "na"] :
        continue

    try :
        att_emb_vectors, att_tokens = biobert.word_vector(att)

        a, b = att_tokens[0], att_emb_vectors[0]

        for l, m in zip(att_tokens[1: ], att_emb_vectors[1: ]) :
            a = a + l
            b += m

        b /= len(att_tokens) 

        unit_emb_vectors, unit_tokens = biobert.word_vector(unit)

        # unit_token = unit_token[0]
        # unit_emb_vector = unit_emb_vector[0]

        a_unit, b_unit = unit_tokens[0], unit_emb_vectors[0]

        if len(unit_tokens) > 1 :
            for l, m in zip(unit_tokens[1: ], unit_emb_vectors[1: ]) :
                a_unit = a_unit + l
                b_unit += m

        b_unit /= len(unit_tokens) 

        b = [k.item() for k in b]
        unit_emb_vector = [k.item() for k in b_unit]

        b.extend(unit_emb_vector)
        a = a.strip() + " " + a_unit

        data = data.append({"word": a.lower().strip(), "text": i, "vectors": b}, ignore_index = True)

    except Exception as e:
        print(e)
        sys.exit(0)# break
        print("cannot prepare embeddings for", text)


words_to_search = ["dose", "age", "treatment", "paracetamol", "enzyme"]


for word_to_search in words_to_search :
    print("Finding vectors related to the word:", word_to_search)
    required_vectors = find_vectors(data, word_to_search)
    print("search_results_of_word: ", word_to_search, ":", required_vectors.shape[0])

    required_vectors = required_vectors.sample(min(15, required_vectors.shape[0]))

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