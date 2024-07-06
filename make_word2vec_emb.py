import csv
import sys
import pandas as pd

from time import time
from datetime import timedelta
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec


PATH = './cancerkg2_numeric_meta_v.txt'

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1

epoch_logger = EpochLogger()

def get_w2v(dataset, embedding_dim=300, min_count=1, window_size=3, num_workers=10, callback=[epoch_logger]):
    w2v = Word2Vec(
        sentences = dataset,
        size = embedding_dim,
        window = window_size,
        min_count = min_count,
        workers = num_workers,
        callbacks=callback
    )

    return w2v


print('Loading Data...')


lines = open(PATH).readlines()

dataset = [line.split() for line in lines]

t0 = time()
word2vec = get_w2v(dataset, min_count=1)
elapsed = time() - t0

print('Elapsed time:', str(timedelta(seconds=elapsed)))

word2vec.save("./embeddings/word2vec/meta_v.word2vec")
