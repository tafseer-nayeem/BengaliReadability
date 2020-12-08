# -*- coding: utf-8 -*-

#!pip uninstall tensorflow

#!pip install tensorflow-gpu==1.14.0

import tensorflow
#tensorflow.__version__

import pandas as pd
import numpy as np
from sklearn.externals import joblib
import seaborn as sn
import matplotlib.pyplot as plt
from keras.preprocessing import *
from keras.layers import *
from keras.models import *
from keras.initializers import *
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import *
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")

tokenizer = joblib.load("/content/drive/My Drive/Complete_dataset/models1/tokenizer_keras.pkl")
vocab_size = joblib.load("/content/drive/My Drive/Complete_dataset/models1/vocab_size.pkl")
train_X_pad_60len = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/train_X_pad_60len.pkl"))
valid_X_pad_60len = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/valid_X_pad_60len.pkl"))
test_X_pad_60len = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/test_X_pad_60len.pkl"))

label_train = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/label_train.pkl"))
label_valid = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/label_valid.pkl"))
label_test = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/label_test.pkl"))

max_length = len(train_X_pad_60len[0])
max_features =vocab_size
embed_size =300
maxlen=max_length

#max_length

"""**Set the path of hdf5 model**"""

f_path ='/content/drive/My Drive/Complete_dataset/models1/kerasmay/bilstmAttention_baseline.hdf5'

checkpoint = ModelCheckpoint(f_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

"""**Bi-LSTM**"""

#f_path

def model_bilstm():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size,trainable = False)(inp)
    x = Bidirectional(CuDNNLSTM(64))(x)
    conc = Dense(64, activation="relu")(x)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

a = model_bilstm()

#a.summary()

a.fit(train_X_pad_60len,label_train,epochs=50, batch_size=16,validation_data=(valid_X_pad_60len, label_valid),callbacks=callbacks_list)

"""**Bilstm + Attention**"""

#f_path

class Attention(Layer):
    def __init__(self, step_dim, W_regularizer=None, b_regularizer=None,W_constraint=None, b_constraint=None,bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2
            hidden = LSTM(64, return_sequences=True)(words)
            sentence = Attention()(hidden)
            # next add a Dense layer (for classification/regression) or whatever...
        """
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(name='{}_W'.format(self.name), shape=(input_shape[-1],),initializer=self.init,regularizer=self.W_regularizer,constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(name='{}_b'.format(self.name), shape=(input_shape[1],),initializer='zero',regularizer=self.b_regularizer,constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        e = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))  # e = K.dot(x, self.W)
        if self.bias:
            e += self.b
        e = K.tanh(e)

        a = K.exp(e)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)

        c = K.sum(a * x, axis=1)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim

def model_bilstm_atten():
    inp = Input(shape=(max_length,))
    x = Embedding(max_features, embed_size, trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Attention(max_length)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

d = model_bilstm_atten()

#d.summary()

d.fit(train_X_pad_60len,label_train,epochs=50, batch_size=16,validation_data=(valid_X_pad_60len, label_valid),callbacks=callbacks_list)

"""**Performance**"""

f_path1 = '/content/drive/My Drive/Complete_dataset/models1/kerasmay/bilstm_baseline.hdf5'
f_path2 = '/content/drive/My Drive/Complete_dataset/models1/kerasmay/bilstmAttention_baseline.hdf5'

h1= model_bilstm()
h1.load_weights(f_path1)

loss, accuracy = h1.evaluate(test_X_pad_60len, label_test)
print('Accuracy:', (accuracy*100))

h2= model_bilstm_atten()
h2.load_weights(f_path2)

loss, accuracy = h2.evaluate(test_X_pad_60len, label_test)
print('Accuracy:', (accuracy*100))

def model_perfomance(model):
  test_result2 = model.predict(test_X_pad_60len)
  test_result22 =[] 
  for i in range(len(test_result2)):
    one =  test_result2[i]
    zero = 1-test_result2[i]
    if one> zero:
      test_result22.append(1)
    else:
      test_result22.append(0)
  
  accu3 =0
  for i in range(len(label_test)):
    if label_test[i] == test_result22[i]:
      accu3 = accu3 +1
  print("accuracy:", (accu3/len(label_test))*100)
  
  predict_label = test_result22
  TP = 0
  TN = 0
  FP = 0
  FN = 0

  for i in range(len(label_test)):
    if label_test[i] ==1 and predict_label[i] ==1:
      TP = TP+1
    if label_test[i] ==0 and predict_label[i] ==1:
      FP = FP+1
    if label_test[i] ==1 and predict_label[i] ==0:
      FN = FN +1
    if label_test[i] ==0 and predict_label[i] ==0:
      TN = TN +1
  print("TP: ",TP)
  print("TN: ",TN)
  print("FP: ",FP)
  print("FN: ",FN)
  P = TP + FN
  N = FP + TN
  print("ACCURACY (USING RULE): ",((TP+TN)/(P+N))*100,"%")
  e1 =(TP/P)*100
  e2 =(TP/(TP+FP))*100
  print("RECALL : ",e1,"%")
  print("PRECISION : ",e2,"%")
  print("F1 measure: ", (2*e1*e2)/(e1+e2),"%" )
  df1 = pd.DataFrame([[TP,FP],[FN,TN]], range(2),range(2))
  ax= plt.subplot()
  sn.heatmap(df1, annot=True, ax = ax,fmt='d')       
  ax.set_xlabel('True labels')
  ax.set_ylabel('Predicted labels')
  ax.set_title('Confusion Matrix')
  ax.xaxis.set_ticklabels([ '1','0'])
  ax.yaxis.set_ticklabels(['1','0'])

#model_perfomance(h1) # bilstm

#model_perfomance(h2) #attention