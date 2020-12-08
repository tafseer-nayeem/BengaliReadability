# -*- coding: utf-8 -*-

import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import text, sequence
from sklearn.externals import joblib

import pandas as pd

import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

trainSet = pd.read_csv('/content/drive/My Drive/Complete_dataset/models1/train.csv')
testSet = pd.read_csv('/content/drive/My Drive/Complete_dataset/models1/test.csv')
validSet = pd.read_csv('/content/drive/My Drive/Complete_dataset/models1/valid.csv')

sentence_train = list(trainSet['sentence'])
sentence_test = list(testSet['sentence'])
sentence_valid = list(validSet['sentence'])


label_train = list(trainSet['class'])
label_test  = list(testSet['class'])
label_valid  = list(validSet['class'])

f1 ="/content/drive/My Drive/Complete_dataset/models1/label_train.pkl"
joblib.dump(label_train,f1)

f2 ="/content/drive/My Drive/Complete_dataset/models1/label_test.pkl"
joblib.dump(label_test,f2)

f3 ="/content/drive/My Drive/Complete_dataset/models1/label_valid.pkl"
joblib.dump(label_valid,f3)

tokenizer = Tokenizer(oov_token="oov")
tokenizer.fit_on_texts(sentence_train + sentence_test + sentence_valid)
train_X = tokenizer.texts_to_sequences(sentence_train)
test_X = tokenizer.texts_to_sequences(sentence_test)
valid_X = tokenizer.texts_to_sequences(sentence_valid)

print(tokenizer.word_index.items())

print(len(tokenizer.word_index))

print(tokenizer.texts_to_sequences(["থdfsdsdfsdfsdতেss"])) #oov

filenamex ="/content/drive/My Drive/Complete_dataset/models1/tokenizer_keras.pkl"
joblib.dump(tokenizer,filenamex)

filenamex1 ="/content/drive/My Drive/Complete_dataset/models1/train_X.pkl"
joblib.dump(train_X,filenamex1)

filenamex2 ="/content/drive/My Drive/Complete_dataset/models1/test_X.pkl"
joblib.dump(test_X,filenamex2)

filenamex3 ="/content/drive/My Drive/Complete_dataset/models1/valid_X.pkl"
joblib.dump(valid_X,filenamex3)

all_sent = sentence_train + sentence_test + sentence_valid

print(len(all_sent))

sentence_length =[]

for i in range(len(all_sent)):
   l = all_sent[i].split(" ")
   sentence_length.append(len(l))

max_length=max(sentence_length)


train_X_pad_60len = pad_sequences(train_X, maxlen=60,padding='post')
test_X_pad_60len = pad_sequences(test_X, maxlen=60,padding='post')
valid_X_pad_60len = pad_sequences(valid_X, maxlen=60,padding='post')

vocab_size = len(tokenizer.word_index)+1

filenamex5="/content/drive/My Drive/Complete_dataset/models1/vocab_size.pkl"
joblib.dump(vocab_size,filenamex5)

#pickle all the padded version

f_path = "/content/drive/My Drive/Complete_dataset/models1/valid_X_pad_60len.pkl"
joblib.dump(valid_X_pad_60len,f_path)