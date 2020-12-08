# -*- coding: utf-8 -*-

#!pip install bpemb
import pandas as pd
import re
import pickle
from sklearn.externals import joblib
from bpemb import BPEmb
import numpy as np
import warnings
warnings.filterwarnings("ignore")

bpemb_25000_300d = BPEmb(lang="bn", dim=300, vs=25000)

def get_embedding_matrix_bpemb(model):
  embedding_matrix = np.zeros((vocab_size,300))
  
  for word, i in tokenizer.word_index.items():
      if word!="oov":
         vect = np.asarray(model.embed(word),dtype='float32')
         if vect is not None:
            vect1 =  np.mean(vect, axis =0, dtype='float32')
            embedding_matrix[i] = vect1
  return embedding_matrix

tokenizer = joblib.load("/content/drive/My Drive/Complete_dataset/models1/tokenizer_keras.pkl")

vocab_size = joblib.load("/content/drive/My Drive/Complete_dataset/models1/vocab_size.pkl")

embd_matrix_bpemb_25000_300d = get_embedding_matrix_bpemb(bpemb_25000_300d)

f_all1 ="/content/drive/My Drive/Complete_dataset/models1/bpemb_models/embd_matrix_bpemb_25000_300d.pkl"
joblib.dump(embd_matrix_bpemb_25000_300d,f_all1)