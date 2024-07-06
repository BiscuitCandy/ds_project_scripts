import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.backend as K
import numpy as np
import csv, sys
from tensorflow import keras
import pandas as pd
from tensorflow.keras.models import Model
from keras.layers.merge import concatenate
from numpy import zeros
from numpy import asarray
from tensorflow.keras.layers import LSTM,Dense,Dropout,Embedding, Concatenate, Input,Lambda , BatchNormalization, Bidirectional

# Loading Chris's Embeddings
from gensim.models import Word2Vec
import gensim
gensim.__version__
csv.field_size_limit(sys.maxsize)
vocab_size = 500000

dfpos=pd.read_csv("../dash_renal/positive_query3_data.csv")
# dfpos = dfpos.sample(min(5000, dfpos.shape[0]))
# dfpos=pd.read_csv(sys.argv[1])
dfpos = dfpos.rename(columns={'Unnamed: 0': 'counter'})
dfneg=pd.read_csv("../dash_renal/negative_query3_data.csv")
# dfneg=pd.read_csv(sys.argv[2])
dfneg = dfneg.rename(columns={'Unnamed: 0': 'counter'})

# df=pd.read_csv("/home/maitry/constrastive/allcommandline/onlycovidtablesckg.csv",delimiter="~")
# df = df.rename(columns={'Unnamed: 0': 'counter'})


# this is manually labeled test data that we want to test on

# for vaccine
# listpos = ['62d6d5b365561b91762c7797','62d6d8f365561b9176309210','62d6d92565561b917630dcfd','62d6d5b965561b91762c7e31','62d6d67465561b91762d30ce', '62d6d7ce65561b91762ee737','62d6d71665561b91762dd560']
# listneg = ['62d6d5bf65561b91762c8650','62d6d61365561b91762cf346','62d6d64d65561b91762cfe29','62d6d65065561b91762d013b','62d6d65565561b91762d07bd','62d6d65d65561b91762d1210','62d6d66365561b91762d1923']

# U HAVE TO CHANGE THIS AS PER YOUR TOPIC: for pregnancy


# listneg = ["640b4daa4dad1ba2d097cc51", "640b4daa4dad1ba2d097cc53", "640b4daa4dad1ba2d097cc4f", "640b4daa4dad1ba2d097cc48", "640b4daa4dad1ba2d097cc46", "640b4daa4dad1ba2d097cc43", "640b4daa4dad1ba2d097cc3d"]
# listpos = ["640b522d4dad1ba2d09fe809", "640b519c4dad1ba2d09e6d00", "640b518c4dad1ba2d09e48a2", "640b51884dad1ba2d09e3f33", "640b51824dad1ba2d09e2ece", "640b51804dad1ba2d09e2bac", "640b517b4dad1ba2d09e202f"]

listneg = ["640b4daa4dad1ba2d097cc3a", "640b4daa4dad1ba2d097cc37", "640b4daa4dad1ba2d097cc4f", "640b4daa4dad1ba2d097cc48", "640b4daa4dad1ba2d097cc46", "640b4daa4dad1ba2d097cc43", "640b4daa4dad1ba2d097cc3d"]
# listpos = ['640b517d4dad1ba2d09e24be','640b51774dad1ba2d09e15b3','640b51744dad1ba2d09e0e41','640b51724dad1ba2d09e0b32','640b51734dad1ba2d09e0bc6', "640b51884dad1ba2d09e3f33", "640b513c4dad1ba2d09da126", "640b51804dad1ba2d09e2bac", "640b517b4dad1ba2d09e202f", "640b4da94dad1ba2d097cb8b", "640b4daa4dad1ba2d097cbef", "640b4daa4dad1ba2d097ccbf"]
listpos = ["640b4da94dad1ba2d097cb8b", "640b4daa4dad1ba2d097ccd9", "640b4daa4dad1ba2d097cd0b", "640b4daa4dad1ba2d097cd2a", "640b4dab4dad1ba2d097ce0b", "640b4dab4dad1ba2d097ceab", "640b4dab4dad1ba2d097cebb"]

TestNegative = dfneg[dfneg.id.isin(listneg)]
TestPositive = dfpos[dfpos.id.isin(listpos)]

dfpos = dfpos.sample(min(2500, dfpos.shape[0]))


RemainingTrainDataPositive = dfpos[~dfpos.id.isin(listpos)]

# print(dfpos.shape)
RemainingTrainDataNegative = dfneg[~dfneg.id.isin(listneg)].sample(min(dfneg.shape[0] - TestNegative.shape[0], RemainingTrainDataPositive.shape[0]))

print("Test Pos Shape: ",TestPositive.shape)
print("Test Neg Shape: ",TestNegative.shape)
print("Train Pos Shape: ",RemainingTrainDataPositive.shape)
print("Train Neg Shape: ",RemainingTrainDataNegative.shape)

# RemainingTrainDataNegative['label']=0
# TestNegative['label']=0

ResultTrain = pd.concat([RemainingTrainDataPositive, RemainingTrainDataNegative])
ResultTest = pd.concat([TestPositive, TestNegative])

ResultTest = shuffle(ResultTest)
ResultTrain = shuffle(ResultTrain)

# def process_res(x):
#     newlist = x.split('@###@')
#     i = 0
#     while i < len(newlist):
#         sublist = newlist[i].split('_')
#         x = "".join([text.lower() for text in sublist if text.isalnum()])
#         newlist[i] = x
#         if len(newlist[i]) == 0:
#             newlist.pop(i)
#             continue
#         i += 1
#     return newlist

# def process_data(x):
#   x=x.replace("[", "")
#   x=x.replace("]", "")
#   x=x.replace("'", "")
#   newlist=x.split(",")
#   for a in range(len(newlist)):
#     newlist[a]=newlist[a].strip()
#     newlist[a]=newlist[a].replace(" ", "_")
#   return "@###@".join(newlist)

# def is_integer(n):
#     try:
#         return "NEG" if (float(n) or int(n)) < 0 else "SMALL_POS" if ((float(n) or int(n)) > 0) and ((float(n) or int(n)) < 1) else "FLOAT" if float(n) else "INT" if int(n) else "ZERO" if float(n)==0.0 else "NOTNUM"
#     except ValueError:
#         return "NOTNUM"
#     else:
#         return "NOTNUM"

# def process_df(df, columns=['data', 'meta_h', 'meta_v'], term_wise=False):
#     df = df.rename(columns={'Unnamed: 0': 'counter'})

#     for col in columns:
#         df[col]=df[col].str.replace(r'\'0.0\'|\'0\'', 'ZERO', regex=True)
#         # df[col] = np.vectorize(process_data)(df[col])
#         df[col] = df[col].apply(process_data)
#         # DONT CHANGE THE ORDER OF THE FOLLOWING LINES OF CODE
#         df[col]=df[col].str.replace(r'([0-9]+\.?[0-9]*(\s)*(\-)(\s)*[0-9]+\.?[0-9]*)', 'RANGE', regex=True)
#         df[col]=df[col].str.replace(r'(-\d*\.?\d*)', 'NEG', regex=True)
#         df[col]=df[col].str.replace(r'(?<![0-9])(0)([.])([0-9]*)', 'SMALLPOS', regex=True)
#         df[col]=df[col].str.replace(r'([0-9]+\.[0-9]*)', 'FLOAT', regex=True)
#         df[col]=df[col].str.replace(r'(?<![a-z|A-Z])(?<![\d.])[0-9]+(?![\d.])(?![a-z|A-Z])', 'INT', regex=True)
#         df[col]=df[col].str.replace(r'%', 'PERCENT', regex=True)
#         df[col]=df[col].str.replace(r'([\w]*(january|february|march|april|may|june|july|august|september|october|november|december|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December)[\w]*)', 'DATE', regex=True)
#         df[col]=df[col].str.replace(r'<', 'LESS', regex=True)
#         df[col]=df[col].str.replace(r'>', 'GREATER', regex=True)
#         df[col]=df[col].str.replace(r'~', 'APPROX', regex=True)
#         df[col]=df[col].str.replace(r'([\w]*(Year|year|time|times|Times|Time|days|Days)[\w]*)', 'UNITTIME', regex=True)
#         df[col]=df[col].str.replace(r'[a-z|\_]*/ml', 'UNITML', regex=True)
#         df[col]=df[col].str.replace(r'[a-z|\_]*/mg', 'UNITMG', regex=True)
#         df[col]=df[col].str.replace(r'[a-z|\_]*/kg', 'UNITKG', regex=True)
#         df[col] = df[col].apply(process_res)
#     return df



# Preprocessing script
def process_res(x):
    newlist = x.split(' ')
    i = 0
    while i < len(newlist):
        sublist = newlist[i].split(' ')
        x = " ".join([text.lower() for text in sublist if text.isalnum()])
        newlist[i] = x
        if len(newlist[i]) == 0:
            newlist.pop(i)
            continue
        i += 1
    return newlist

def process_data(x):
  x = str(x)
  x=x.replace("[", "")
  x=x.replace("]", "")
  x=x.replace("'", "")
  newlist=x.split(",")

  for a in range(len(newlist)):
    newlist[a]=newlist[a].strip()
    # newlist[a]=newlist[a].replace(" ", "_")
  return " ".join(newlist)


def process_df(df, columns=['data', 'meta_h', 'meta_v'], term_wise=False):
    df = df.rename(columns={'Unnamed: 0': 'counter'})

    for col in columns:
        # df[col]=df[col].str.replace(r'\'0.0\'|\'0\'', 'ZERO', regex=True)
        # df[col] = np.vectorize(process_data)(df[col])
        df[col] = df[col].apply(process_data)
        # # DONT CHANGE THE ORDER OF THE FOLLOWING LINES OF CODE
        # df[col]=df[col].str.replace(r'([0-9]+\.?[0-9]*(\s)*(\-)(\s)*[0-9]+\.?[0-9]*)', 'RANGE', regex=True)
        # df[col]=df[col].str.replace(r'(-\d*\.?\d*)', 'NEG', regex=True)
        # df[col]=df[col].str.replace(r'(?<![0-9])(0)([.])([0-9]*)', 'SMALLPOS', regex=True)
        # df[col]=df[col].str.replace(r'([0-9]+\.[0-9]*)', 'FLOAT', regex=True)
        # df[col]=df[col].str.replace(r'(?<![a-z|A-Z])(?<![\d.])[0-9]+(?![\d.])(?![a-z|A-Z])', 'INT', regex=True)
        # df[col]=df[col].str.replace(r'%', 'PERCENT', regex=True)
        # df[col]=df[col].str.replace(r'([\w]*(january|february|march|april|may|june|july|august|september|october|november|december|JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|January|February|March|April|May|June|July|August|September|October|November|December)[\w]*)', 'DATE', regex=True)
        # df[col]=df[col].str.replace(r'<', 'LESS', regex=True)
        # df[col]=df[col].str.replace(r'>', 'GREATER', regex=True)
        # df[col]=df[col].str.replace(r'~', 'APPROX', regex=True)
        # df[col]=df[col].str.replace(r'([\w]*(Year|year|time|times|Times|Time|days|Days)[\w]*)', 'UNITTIME', regex=True)
        # df[col]=df[col].str.replace(r'[a-z|\_]*/ml', 'UNITML', regex=True)
        # df[col]=df[col].str.replace(r'[a-z|\_]*/mg', 'UNITMG', regex=True)
        # df[col]=df[col].str.replace(r'[a-z|\_]*/kg', 'UNITKG', regex=True)
        df[col] = df[col].apply(process_res)
    return df

# df = process_df(df)

ResultTest = process_df(ResultTest)

ResultTrain = process_df(ResultTrain)

def tokenizeData(decodedList, max_size=190,t = None):
    # create the tokenizer
    if not t:
      t = Tokenizer(num_words=50000, filters='\'')
      t.fit_on_texts(decodedList)
    # integer encode documents
    numbered_docs = t.texts_to_sequences(decodedList)
    data = numbered_docs

    word_index = t.word_index
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown

    word_index["<UNUSED>"] = 3


    data = tf.keras.preprocessing.sequence.pad_sequences(data,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=max_size)
    return data, t

#Tokenizing for data

train_data, tok = tokenizeData(list(ResultTrain['data']), max_size=100)
test_data, toktest = tokenizeData(list(ResultTest['data']),t = tok, max_size=100)


# #Tokenizing for attributes
# train_attr, tokattr = tokenizeData(list(ResultTrain['meta_h']), max_size=100)
# test_attr, toktestattr = tokenizeData(list(ResultTest['meta_h']),t = tokattr, max_size=100)

# Loading Chris's Embeddings
from gensim.models import Word2Vec


# attr_word_index = tokattr.word_index
# attr_word_index = {k:(v+3) for k,v in attr_word_index.items()}
# attr_word_index["<PAD>"] = 0
# attr_word_index["<START>"] = 1
# attr_word_index["<UNK>"] = 2  # unknown

# attr_word_index["<UNUSED>"] = 3


data_word_index = tok.word_index

data_word_index = {k:(v+3) for k,v in data_word_index.items()}
data_word_index["<PAD>"] = 0
data_word_index["<START>"] = 1
data_word_index["<UNK>"] = 2  # unknown

data_word_index["<UNUSED>"] = 3

# attr_embeddings_index = {}

# # word2Vec = Word2Vec.load('/home/maitry/constrastive/embeddingmodels/terms/metadata/newtrain.md.cord19_w2v_preproc.model')
# # word2Vec = Word2Vec.load('/home/maitry/constrastive/embeddingmodels/terms/metadata/newtrain.md.cord19_w2v_preproc.model')
# word2Vec = Word2Vec.load(sys.argv[3])
# for word in word2Vec.wv.key_to_index:
#    attr_embeddings_index[word] = word2Vec.wv[word]

data_embeddings_index = {}

word2VecData = Word2Vec.load("../full_ckg/data.terms.ckg_3_all_w2v_preproc.model")
# word2Vec = Word2Vec.load('/home/maitry/constrastive/embeddingmodels/terms/metadata/newtrain.md.cord19_w2v_preproc.model')
# word2VecData = Word2Vec.load(sys.argv[4])
for word in word2VecData.wv.key_to_index:
    data_embeddings_index[word] = word2VecData.wv[word]



# word_index_len = len(data_word_index)+len(attr_word_index)+2

# embedding_matrix_attr = np.zeros((len(attr_word_index) + 1, 300))
embedding_matrix_data = np.zeros((len(data_word_index) + 1, 300))

print('Found %s word vectors.' % len(data_embeddings_index))
print('Len of data_word_index', len(data_word_index))

# for word, i in attr_word_index.items():
#     embedding_vector = attr_embeddings_index.get(word)
#     #print(attr_embeddings_index.get(word))
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix_attr[i] = embedding_vector

for word, i in data_word_index.items():
    embedding_vector = data_embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix_data[i] = embedding_vector



def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(y_true)
    recall = true_positives / (possible_positives + K.epsilon())

    return recall

def create_model(data_input,label_input):
# def create_model(attr_input,data_input,label_input):
#   layer1 = tf.keras.layers.Embedding(len(embedding_matrix_attr),
#                           300,
#                           weights=[embedding_matrix_attr],
#                           input_length=300,
#                           trainable=False)(attr_input)
  layer2 = tf.keras.layers.Embedding(len(embedding_matrix_data),
                          300,
                          weights=[embedding_matrix_data],
                          input_length=300,
                          trainable=False)(data_input)
  layer2 = tf.keras.layers.Conv1D(filters = 256, kernel_size=3)(layer2)
#   layer = tf.keras.layers.Concatenate()([layer1,layer2])
  # layer = tf.keras.layers.Concatenate()([layer2])
  layer = tf.keras.layers.GlobalAveragePooling1D()(layer2)
  layer = tf.keras.layers.Dense(64, activation='relu')(layer)
  layer = tf.keras.layers.Dense(1, activation='sigmoid')(layer)
#   model = Model(inputs=[attr_input,data_input],outputs=layer)
  model = Model(inputs=[data_input],outputs=layer)
  return model

import datetime
log_dir = "logs/fit/"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def train_model(model, x_train, y_train, x_test, y_test, bz=512, ep=1, v=1):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', precision, recall])
    history = model.fit(x_train, y_train, batch_size=bz, verbose=v, callbacks=[early_stop,tensorboard_callback], validation_data=(x_test, y_test),validation_steps = int(1371), use_multiprocessing=False , workers=1)
    return history


# attr_input = tf.keras.Input(shape=(100,),name='attr_input')
data_input = tf.keras.Input(shape=(100,),name='data_input')
label_input = tf.keras.Input(shape=(),name='label_input')

# model = create_model(attr_input,data_input,label_input)
model = create_model(data_input,label_input)
model.summary()

# x_test = train_data[::10]
x_train_data = np.delete(train_data, np.arange(0, len(train_data), 1000), axis=0)
# x_train_attr = np.delete(train_attr, np.arange(0, len(train_attr), 1000), axis=0)
labels = np.array(ResultTrain['label'])
y_test = labels[::1000]
y_train = np.delete(labels, np.arange(0, labels.size, 1000))

history = train_model(model, [x_train_data], y_train, [train_data[::1000]], y_test, ep=1, v=1)


# history = train_model(model, [x_train_attr,x_train_data], y_train, [train_attr[::1000],train_data[::1000]], y_test, ep=1, v=1)


from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict([test_data])
# y_pred = model.predict([test_attr,test_data])
y_pred  = np.where(y_pred > 0.5, 0, 1)

print(classification_report(ResultTest['label'].values,y_pred))
