# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from sklearn.externals import joblib

#!pip install laserembeddings

#!python -m laserembeddings download-models

from laserembeddings import Laser

laser = Laser()

tokenizer1 = joblib.load("/content/drive/My Drive/Complete_dataset/models1/tokenizer_keras.pkl")

vocab_size1 = joblib.load("/content/drive/My Drive/Complete_dataset/models1/vocab_size.pkl")

embedding_matrix_laser = np.zeros((vocab_size1,1024))
  
for word, i in tqdm(tokenizer1.word_index.items()):
      if word!="oov":
         vect = np.asarray(list(laser.embed_sentences([x],lang='bn')), dtype='float32')[0]
         if vect is not None:
            embedding_matrix_laser[i] = vect



filenamey4 ="/content/drive/My Drive/Complete_dataset/models1/embd_matrix_laser.pkl"
joblib.dump(embedding_matrix_laser,filenamey4)