# -*- coding: utf-8 -*-

import pandas as pd
import re
import numpy as np

from sklearn.externals import joblib
from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

tokenizer = joblib.load("/content/drive/My Drive/Complete_dataset/models1/tokenizer_keras.pkl")
vocab_size = joblib.load("/content/drive/My Drive/Complete_dataset/models1/vocab_size.pkl")

def get_embedding_matrix_w2v(model,dim):
  embedding_matrix = np.zeros((vocab_size, dim))
  
  for word, i in tqdm(tokenizer.word_index.items()):
      if word!="oov" and word in list(model.wv.vocab) :
         
         embedding_matrix[i] = np.asarray(list(model[word]), dtype='float32')

  return embedding_matrix



"""**Word2vec**"""

model_path = "/content/drive/My Drive/Complete_dataset/models1/bengali_word2vec.model"

w2v_model = Word2Vec.load(model_path)

embd_matrix_word2vec = get_embedding_matrix_w2v(w2v_model,300)

print(embd_matrix_word2vec)

filenamey1 ="/content/drive/My Drive/Complete_dataset/models1/embd_matrix_word2vec.pkl"
joblib.dump(embd_matrix_word2vec,filenamey1)

"""**gloVe**"""

glove_path = "/content/drive/My Drive/Complete_dataset/models1/bn_glove.39M.300d.txt"

glove_model = {}

with open(glove_path, 'r') as f:

  for line in tqdm(f):
      values = line.split()
      word = values[0]
      vector = np.asarray(values[1:], "float32")
      glove_model[word] = vector

def get_embedding_matrix_glove(model,dim):
  embedding_matrix = np.zeros((vocab_size, dim))
  
  for word, i in tqdm(tokenizer.word_index.items()):
      if word!="oov" and word in list(model.keys()) :
         
         embedding_matrix[i] = model[word]

  return embedding_matrix

embd_matrix_glove = get_embedding_matrix_glove(glove_model, 300)

print(embd_matrix_glove)

filenamey2 ="/content/drive/My Drive/Complete_dataset/models1/embd_matrix_glove.pkl"
joblib.dump(embd_matrix_glove,filenamey2)