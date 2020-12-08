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

f_path ='/content/drive/My Drive/Complete_dataset/models1/kerasmay/bilstmPooling.hdf5'

checkpoint = ModelCheckpoint(f_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

"""**Bi-LSTM** **+ Pooling**"""

def model_lstm_pooling():
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size,trainable = False)(inp)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

u1 = model_lstm_pooling()

#u1.summary()

u1.fit(train_X_pad_60len,label_train,epochs=50, batch_size=16,validation_data=(valid_X_pad_60len, label_valid),callbacks=callbacks_list)

"""**PERFORMANCE**"""

h28= model_lstm_pooling()
h28.load_weights(f_path)

loss, accuracy = h28.evaluate(test_X_pad_60len, label_test)
print('Accuracy:'," ", (accuracy*100))

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

print(model_perfomance(h28))

