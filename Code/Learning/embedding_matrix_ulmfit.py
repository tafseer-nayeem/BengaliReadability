# -*- coding: utf-8 -*-

#!pip install sentencepiece

import pandas as pd
import re
import numpy as np
from sklearn.externals import joblib
import numpy as np
from fastai.text import *
import sentencepiece as spm
import pdb
import fastai, torch
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#!pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
#!pip install inltk

sp = spm.SentencePieceProcessor()
sp.Load("/content/drive/My Drive/Complete_dataset/models1/fastai_models1/bengali_lm.model")
itos = [sp.IdToPiece(int(i)) for i in range(30000)]

embeddings_ulmfit = pd.read_csv("/content/drive/My Drive/Complete_dataset/models1/embeddings_ulmfit.tsv", sep='\t', header =None)

#embeddings_ulmfit

embeddings_ulmfit2 =[]

for i in range(len(embeddings_ulmfit)):
          embeddings_ulmfit2.append(list(embeddings_ulmfit.iloc[i]))

print(embeddings_ulmfit2[1])

tokenizer = joblib.load("/content/drive/My Drive/Complete_dataset/models1/tokenizer_keras.pkl")
vocab_size = joblib.load("/content/drive/My Drive/Complete_dataset/models1/vocab_size.pkl")

from inltk.inltk import setup

setup('bn')

from inltk.inltk import get_embedding_vectors
from inltk.inltk import tokenize

def get_embedding_vectors1(word):
    word_token = tokenize(word,'bn')
    embd_vec =[]
    for i in range(len(word_token)):
      if word_token[i] in itos:
        indx = itos.index(word_token[i])
        embd_vec.append(embeddings_ulmfit2[indx])
    return embd_vec

embd_matrix_inltk_ulmfit_400d = np.zeros((vocab_size,400))

for word, i in tqdm(tokenizer.word_index.items()):
      if word!="oov":
         vect = np.asarray(get_embedding_vectors1(word), dtype='float32')
         if vect is not None:
            vect1 =  np.mean(vect, axis =0, dtype='float32')
            embd_matrix_inltk_ulmfit_400d[i] = vect1

#embd_matrix_inltk_ulmfit_400d

#embd_matrix_inltk_ulmfit_400d.shape

fpath = "/content/drive/My Drive/Complete_dataset/models1/inltkmodel/embd_matrix_inltk_ulmfit_400d.pkl"

joblib.dump(embd_matrix_inltk_ulmfit_400d,fpath)