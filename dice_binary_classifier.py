import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Embedding

data = pd.read_csv("meta_v_dice_labels2.tildesv", sep="~")

# data = data.sample(50)

data = data[(data["Rx1"] < 1000000) & (data["Ry1"] < 1000000) & (data["Rx2"] < 1000000) & (data["Ry2"] < 1000000)]

y = data["label"]

X = data[["Rx1", "Ry1", "Rx2", "Ry2"]]

print(X.shape, y.shape)

# Define the model
model = Sequential()

# Flatten the input features
model.add(Flatten(input_shape=(4,)))

# Embedding layer with size 339
model.add(Embedding(input_dim=1000000, output_dim=339))

# Flatten the embedding layer
model.add(Flatten())

# Dense layers for classification
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))  # Output layer with 10 classes (digits 0-9)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Split the dataset into training and testing sets

# Train the model
model.fit(X, y, epochs=2, batch_size=32)

model.save("dice_binary_classifier.model")