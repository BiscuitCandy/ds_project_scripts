import tensorflow as tf
from transformers import AutoModel, AutoTokenizer

model_name = "dmis-lab/biobert-base-cased-v1.1"

# Load BioBERT model and tokenizer
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

input_text = open("./raw_biobert_data").readlines()

encoded_inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")

# Forward pass through the model
outputs = model(**encoded_inputs)
word_embeddings = outputs.last_hidden_state

# Save word embeddings as TensorFlow tensors
tf.saved_model.save(word_embeddings, "data_biobert_embeddings")