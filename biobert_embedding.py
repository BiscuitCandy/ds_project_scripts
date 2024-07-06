import os
import torch
import logging
import numpy as np
from pathlib import Path
import csv
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(filename='app.log', filemode='w',format='%(asctime)s %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)

meta_ = open("./embeddings_real/cancerkg_real_meta_v_md.tsv", 'w')
data_ = open("./embeddings_real/cancerkg_real_meta_v.tsv", 'w')
meta2_ = open("./embeddings_numerics/cancerkg_numeric_meta_v_md.txt", 'w')
data2_ = open("./embeddings_numerics/cancerkg_numeric_meta_v.tsv", 'w')

data_writer = csv.writer(data_, delimiter = '\t')
meta_writer = csv.writer(meta_, delimiter='\t')

data_writer2 = csv.writer(data2_, delimiter = '\t')
meta_writer2 = csv.writer(meta2_, delimiter='\t')

meta_writer.writerow(['word', 'text'])

class BiobertEmbedding(object):
    def __init__(self, model_path="../biobert/new_biobert2_numeric_64/meta_v"):
        if model_path is not None:
            self.model_path = model_path
        else:
            # self.model_path = downloader.get_BioBert("google drive")
            print("given model_path is invalid")

        self.tokens = ""
        self.sentence_tokens = ""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path)
        logger.info("Initialization Done !!")

    def process_text(self, text):

        marked_text = "[CLS] " + text + " [SEP]"
        # Tokenize our sentence with the BERT tokenizer.
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

        # Mark each of the tokens as belonging to sentence "1".
        segments_ids = [1] * len(tokenized_text)
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()
        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

        return np.array(encoded_layers)


    def word_vector(self, text, handle_oov=True, filter_extra_tokens=True):

        tokenized_text = self.process_text(text)
        encoded_layers = self.eval_fwdprop_biobert(tokenized_text)
        encoded_layers = torch.split(torch.from_numpy(encoded_layers).view(-1), encoded_layers.shape[1])

        token_embeddings = torch.stack(encoded_layers, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        # print(token_embeddings.shape)
        # Swap dimensions 0 and 1.
        token_embeddings = torch.transpose(token_embeddings, 0, 1)
        token_embeddings.reshape((*token_embeddings.shape, 1))

        self.tokens = tokenized_text[1:-1]

        if handle_oov:
            self.tokens, token_embeddings = self.handle_oov(self.tokens, token_embeddings)
        
        # ignore_words = ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i', 'be', 'have', 'do', 'or', 'not', 'this', 'this', 'but', 'from', 'by', 'at', 'as', 'we', 'an', 'will', 'can', 'up', 'so', 'if', 'go', 'no', 'see', 'my', 'me', 'like', 'on', 'our', 'him', 'all', 'your', 'her', 'out', 'when', 'get', 'about', 'would', 'into', 'up', 'there', 'could', 'other', 'than', 'some', 'upon', 'them', 'only', 'then', 'these', 'also', 'now']

        flag = 0
        x1 = y1 = 0
        for i, j in zip(self.tokens, token_embeddings) :
            # print(i)
            x = j
            y = i
            if "[" == i :
                flag += 1
                x1 = j
                y1 = i
                continue
            elif i =="]":
                x1 += j
                y1 += i
                x1 /= flag
                x = x1 
                y = y1
                flag = 0
                print(y)
                data_writer2.writerow(np.array(x))
                meta_writer2.writerow(y.strip())

            elif flag > 0 :
                x1 += j
                y1 += i
                flag += 1
                continue
            
            # print(y, "*********")
            data_writer.writerow(np.array(x))
            meta_writer.writerow([y, text.strip()])

        return token_embeddings


if __name__ == "__main__":

    biobert = BiobertEmbedding()

    iu = 0

    with open("../cancerkg2-numeric_meta_v.txt") as f:
        texts = f.readlines()

    for tex in texts :
        text = tex.replace(",", " ")[:64]
        try: word_embeddings = biobert.word_vector(text)
        except: continue