#! /bin/
import os
import torch
import logging
import numpy as np
from pathlib import Path
import h5py
from transformers import BertModel, BertTokenizer
import argparse

logging.basicConfig(filename='app.log', filemode='w',format='%(asctime)s %(message)s', level=logging.INFO)

class BiobertEmbedding(object):
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='make_embeddings.py',
                    usage='make_embeddings.py --input_file INPUT_FILE --output_vector_file OUTPUT_VECTOR_FILE --output_context_file OUTPUT_METADATA_FILE',
                    description='Create biobert embeddings')

    parser.add_argument('--model', help="specify one of the following [DATA, META_V, META_H] based on the input data")
    parser.add_argument('--input_file', help='Input_file_path')
    parser.add_argument('--output_file', help='Outputfile Stores the output of the embeddings')
    # parser.add_argument('--output_context_file', help='output_metadata_file_path')

    args = parser.parse_args()

    if (not args.model) or (not args.input_file) or (not args.output_file) :
        parser.print_help()
        exit(0)

    if not os.path.isfile(args.input_file):
        print("No  file named", args.input_file, "exists")
        parser.print_help()
        exit(0)

    model_path = f"../models/cancerkg_new_biobert_seq_64/{args.model.lower()}"

    biobert = BiobertEmbedding(model_path=model_path)

    with open(args.input_file) as f:
        texts = f.readlines()[5:10]

    # meta_ = open(args.output_context_file, 'w+')
    # data_ = open(args.output_vector_file, 'w+')

    # data_writer = csv.writer(data_, delimiter = '\t')
    # meta_writer = csv.writer(meta_, delimiter='\t')

    file = h5py.File(args.output_file, 'w')

    # meta_writer.writerow(['word', 'text'])

    for tex in texts :
        text = tex.replace(",", " ").strip()
        try :
            word_embeddings, tokens = biobert.word_vector(text)

            # for i, j in zip(word_embeddings, tokens):
            #         data_writer.writerow([tensor.item() for tensor in i])
            #         meta_writer.writerow([j, text.strip()])

        except:
            print("An exception occurred at text", text)

        sentence_group = file.create_group(text.strip())

        for i, j in zip(tokens, word_embeddings) :
            # print(text.strip(), i)
            word = sentence_group.create_group(i)
            j_new = np.array([j_j.item() for j_j in j]).reshape((768, 1))
            word.attrs['vector'] = j_new


    print("Com[ple]ted")