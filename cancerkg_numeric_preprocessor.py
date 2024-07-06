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
from transformers import BertTokenizer
import re

df = pd.read_csv("./full_cancerkg.csv", sep = "~").sample(5)
# df = df.sample(50)

def convert_to_range_with_std_dev(number, std_dev_percentage=10):
    num = float(number)  # Convert the number to a floating-point number
    std_dev = abs((num if num else 1) * std_dev_percentage / 100)
    lower_bound = num - std_dev
    upper_bound = num + std_dev
    return f"[{lower_bound:.1f}|{upper_bound:.1f}]"

def convert_text_to_ranges(text, std_dev_percentage=10):
    # Regular expression to match individual numbers (positive or negative, integer or floating-point) not part of words
    number_pattern = r'(?<![a-zA-Z])\b(?<![0-9.-])-?\d+(\.\d+)?(?![0-9.-])\b(?![a-zA-Z])'
    # Regular expression to match numbers inside ranges (e.g., 1-5, 10-20, etc.)
    range_pattern = r'\b(\d+(\.\d+)?)-(\d+(\.\d+)?)\b'

    def process_number(match):
        return convert_to_range_with_std_dev(float(match.group()), std_dev_percentage)

    # Replace individual numbers with ranges (within the specified standard deviation)
    result = re.sub(number_pattern, process_number, text)

    # Replace numbers inside existing ranges with floating-point numbers
    def process_range(match):
        lower_bound = float(match.group(1))
        upper_bound = float(match.group(3))
        return f"{lower_bound:.1f}-{upper_bound:.1f}"

    result = re.sub(range_pattern, process_range, result)

    return result

def process_res(x):
    #print("*", x)
    newlist = x.split(' ')
    i = 0
    while i < len(newlist):
        sublist = newlist[i].split(' ')
        x = " ".join([text.lower() for text in sublist if text.isalnum()][:300])
        newlist[i] = x
        if len(newlist[i]) == 0:
            newlist.pop(i)
            continue
        i += 1
        #print(newlist)
    return ",".join(newlist)

def process_data(x):
  x = str(x)
  x=x.replace("[", "")
  x=x.replace("]", "")
  x=x.replace("'", "")
  newlist=x.split(",")

  for a in range(len(newlist)):
    newlist[a]=newlist[a].strip()
  return " ".join(newlist)


def process_df(df, columns=['data', 'meta_h', 'meta_v'], term_wise=True):
    df = df.rename(columns={'Unnamed: 0': 'counter'})

    for col in columns:
        df[col] = df[col].apply(process_data)
        
        # DONT CHANGE THE ORDER OF THE FOLLOWING LINES OF CODE
        
        # df[col]=df[col].str.replace(r'\'0.0\'|\'0\'', 'ZERO', regex=True)
        # df[col]=df[col].str.replace(r'\d+(\.\d+)?\s*\[\s*\d+(\.\d+)?\s*to\s*\d+(\.\d+)?\s*\]', 'RANGE', regex=True)
        # df[col]=df[col].str.replace(r'-\d+(\.\d+)?', '', regex=True)
        df[col]=df[col].str.replace('%', ' PERCENT', regex=True)
        # df[col]=df[col].str.replace(r'0(\.\d+)?', '', regex=True)
        # df[col]=df[col].str.replace(r'\d+(\.\d+)?', '', regex=True)
        # df[col]=df[col].str.replace(r'(?<![a-z|A-Z])(?<![\d.])[0-9]+(?![\d.])(?![a-z|A-Z])', '', regex=True)
        df[col]=df[col].str.replace(r'\d{1,2}/\d{1,2}/\d{4}', ' DATE ', regex=True)
        df[col]=df[col].str.replace(r'<', ' LESS ', regex=True)
        df[col]=df[col].str.replace(r'>', ' GREATER ', regex=True)
        df[col]=df[col].str.replace(r'~', ' APPROX ', regex=True)
        df[col]=df[col].str.replace(r'/mg\b', ' UNITMG ', regex=True)
        df[col]=df[col].str.replace(r'/kg\b', ' UNITKG ', regex=True)
        df[col]=df[col].str.replace(r'/ml\b', ' UNITML ', regex=True)
        df[col]=df[col].str.replace(r'/L\b', ' UNITL ', regex=True)
        # df[col]=df[col].str.replace(r'([\w]*(Year|year|time|times|Times|Time|days|Days)[\w]*)', 'UNITTIME', regex=True)
        df[col]=df[col].str.replace(r'[a-z|\_]*/ml', ' UNITML ', regex=True)
        df[col]=df[col].str.replace(r'[a-z|\_]*/mg', ' UNITMG ', regex=True)
        df[col]=df[col].str.replace(r'[a-z|\_]*/kg', ' UNITKG ', regex=True)
        # print(df[col])

        df[col] = df[col].apply(process_res)
        # print(df[col])
    return df

df = process_df(df)

df = df[["meta_v", "meta_h", "data"]]

# df.to_csv("../cancerkg/tokenized_cancerkg.csv", sep="~")

for i in ["data", "meta_h", "meta_v"] :
    print("Start:", i)
    data = df[[i]]
    data = data.dropna()
    data = data.drop_duplicates()

    f = open(f"cancerkg2_numeric_{i}.txt", 'w+')

    for j in data[i] :
        if "nan" in str(j) or str(j).strip() == "" : continue
        j_processed = convert_text_to_ranges(j, std_dev_percentage=10)
        f.write(str(j_processed) + "\n")
        # print(j_processed)

    print("Done:", i)

print("Com[ple]ted")