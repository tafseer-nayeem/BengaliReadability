# -*- coding: utf-8 -*-
#!pip install fasttext
import fasttext

import pandas as pd
import re
import numpy as np
from numpy import array

from sklearn.externals import joblib

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

tokenizer = joblib.load("/content/drive/My Drive/Complete_dataset/models1/tokenizer_keras.pkl")

vocab_size = joblib.load("/content/drive/My Drive/Complete_dataset/models1/vocab_size.pkl")

def get_embedding_matrix(model,dim):
  embedding_matrix = np.zeros((vocab_size, dim))
  
  for word, i in tokenizer.word_index.items():
      if word!="oov":
         vect = np.asarray(list(model[word]), dtype='float32')
         if vect is not None:
            embedding_matrix[i] = vect
  return embedding_matrix



"""**Fasttext 300 dimensional pretrained**"""

#!wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.bin.gz

#!gunzip cc.bn.300.bin.gz

model_fasttext_pretrained = fasttext.load_model("cc.bn.300.bin")

embd_matrix_fasttext_preTrained = get_embedding_matrix(model_fasttext_pretrained,300)


filenamey4 ="/content/drive/My Drive/Complete_dataset/models1/embd_matrix_fasttext_preTrained.pkl"
joblib.dump(embd_matrix_fasttext_preTrained,filenamey4)

