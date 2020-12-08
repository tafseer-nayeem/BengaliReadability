# -*- coding: utf-8 -*-
#!pip install bert-for-tf2

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.externals import joblib
import tensorflow as tf
import tensorflow_hub as hub
import bert

def get_model(model_url, max_seq_length):
  labse_layer = hub.KerasLayer(model_url, trainable=True)

  # Define input.
  input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
  input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
  segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

  # LaBSE layer.
  pooled_output,  _ = labse_layer([input_word_ids, input_mask, segment_ids])

  # The embedding is l2 normalized.
  pooled_output = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(pooled_output)

  # Define model.
  return tf.keras.Model(
        inputs=[input_word_ids, input_mask, segment_ids],
outputs=pooled_output), labse_layer

max_seq_length = 64
labse_model, labse_layer = get_model(model_url="https://tfhub.dev/google/LaBSE/1", max_seq_length=max_seq_length)

vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)

def create_input(input_strings, tokenizer, max_seq_length):
  
   input_ids_all, input_mask_all, segment_ids_all = [], [], []
   
   for input_string in input_strings:
     input_tokens = ["[CLS]"] + tokenizer.tokenize(input_string) + ["[SEP]"]
     input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
     sequence_length = min(len(input_ids), max_seq_length)
     
     
     if len(input_ids) >= max_seq_length:
       input_ids = input_ids[:max_seq_length]
     else:
  
       input_ids = input_ids + [0] * (max_seq_length - len(input_ids))
     
     input_mask = [1] * sequence_length + [0] * (max_seq_length - sequence_length)
     input_ids_all.append(input_ids)
     input_mask_all.append(input_mask)
     segment_ids_all.append([0] * max_seq_length)
   return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)


def encode(input_text):
   input_ids, input_mask, segment_ids = create_input(input_text, tokenizer, max_seq_length)
   return labse_model([input_ids, input_mask, segment_ids])

tokenizer1 = joblib.load("/content/drive/My Drive/Complete_dataset/models1/tokenizer_keras.pkl")

vocab_size1 = joblib.load("/content/drive/My Drive/Complete_dataset/models1/vocab_size.pkl")

embedding_matrix_labse = np.zeros((vocab_size1,768))
  
for word, i in tqdm(tokenizer1.word_index.items()):
      if word!="oov":
         vect = np.asarray(list(encode([word])), dtype='float32')[0]
         if vect is not None:
            embedding_matrix_labse[i] = vect



filenamey4 ="/content/drive/My Drive/Complete_dataset/models1/embd_matrix_labse.pkl"
joblib.dump(embedding_matrix_labse,filenamey4)