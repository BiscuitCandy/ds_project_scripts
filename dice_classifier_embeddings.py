import numpy as np
import pandas as pd
import csv
from tensorflow.keras.models import Model, load_model

limit = 1000000

data_file = open("dice_emebddings.psv", "w+")
writer = csv.writer(data_file, delimiter="|")

model = load_model("dice_binary_classifier.model")
embedding_layer = Model(inputs=model.input, outputs=model.layers[1].output)

data = pd.read_csv("meta_v_dice_labels2.tildesv", sep="~")

data = data[(data["Rx1"] < limit) & (data["Ry1"] < limit) & (data["Rx2"] < limit) & (data["Ry2"] < limit)]

X = data[["Rx1", "Ry1", "Rx2", "Ry2"]]

for i in range(81570) :
    input_data =  X.iloc[i:i+1, :]
    embeddings = embedding_layer.predict(input_data)
    rx1_v = embeddings[0][0].tolist()
    ry1_v = embeddings[0][1].tolist()

    writer.writerow([input_data["Rx1"].values[0], input_data["Ry1"].values[0], *rx1_v, *ry1_v])
    