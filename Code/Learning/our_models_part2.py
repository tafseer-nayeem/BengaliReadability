# -*- coding: utf-8 -*-
#!pip uninstall tensorflow

#!pip install tensorflow-gpu==1.14.0

import tensorflow
#tensorflow.__version__


#!pip install keras==2.3.1

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

train_juk1 = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/train_juk.pkl"), dtype = 'float32')
test_juk1 = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/test_juk.pkl"), dtype = 'float32')
valid_juk1 = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/valid_juk.pkl"), dtype = 'float32')


train_strlen1 = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/strlen_train.pkl"), dtype = 'float32')
test_strlen1 = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/strlen_test.pkl"), dtype = 'float32')
valid_strlen1 = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/strlen_valid.pkl"), dtype = 'float32')

train_strlen= train_strlen1.reshape(len(train_strlen1),1)
test_strlen = test_strlen1.reshape(len(test_strlen1),1)
valid_strlen = valid_strlen1.reshape(len(valid_strlen1),1)


train_juk= train_juk1.reshape(len(train_juk1),1)
test_juk = test_juk1.reshape(len(test_juk1),1)
valid_juk = valid_juk1.reshape(len(valid_juk1),1)

"""**embed_size**

*word2vec = 300*

*glove = 300*

*fasttext = 300*

*inltk ulmfit = 400*

*inltk transformerXL = 410* 

*BPEmb = 300*

*labse = 768*

*laser = 1024*
"""

max_length = len(train_X_pad_60len[0])
max_features =vocab_size
embed_size =300
maxlen=max_length

embedding_matrix = np.array(joblib.load("/content/drive/My Drive/Complete_dataset/models1/embd_matrix_glove.pkl"))

"""**Set the path of hdf5 model**"""

f_path ='/content/drive/My Drive/Complete_dataset/models1/kerasmay/laser_all.hdf5'

checkpoint = ModelCheckpoint(f_path, monitor='val_loss', verbose=1, save_best_only=True,mode='min')
callbacks_list = [checkpoint]

"""**BiLSTM + Pooling + no additional**"""

def model_lstm_pooling_noadd(embedding_matrix):
    inp = Input(shape=(maxlen,),name='sequnece_input')
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = True)(inp)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

c2= model_lstm_pooling_noadd(embedding_matrix)

c2.fit(train_X_pad_60len,label_train,epochs=50, batch_size=16,validation_data=(valid_X_pad_60len, label_valid),callbacks=callbacks_list)

"""**BiLSTM + Pooling + character/string length (CL) + juk (CC)**"""

def model_lstm_pooling_strlenjuk(embedding_matrix):
    inp = Input(shape=(maxlen,), name='sequnece_input')
    strlen_input = Input(shape=(1,), name='string_length_input')
    juk_input = Input(shape=(1,), name='juk_input')
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = True)(inp)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = concatenate([conc,strlen_input])
    conc = concatenate([conc,juk_input])
    conc = Dense(64, activation="relu")(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    model = Model(inputs=[inp, strlen_input, juk_input], outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

d7 = model_lstm_pooling_strlenjuk(embedding_matrix)

d7.fit([train_X_pad_60len, train_strlen, train_juk],label_train,epochs=50, batch_size=16,validation_data=([valid_X_pad_60len, valid_strlen, valid_juk], label_valid),callbacks=callbacks_list)

"""**PERFORMANCE**"""

f_path_noadditional = "/content/drive/My Drive/Complete_dataset/models1/kerasmay/glove_noadditional.hdf5"
f_path_all = "/content/drive/My Drive/Complete_dataset/models1/kerasmay/glove_all.hdf5"

x1= model_lstm_pooling_noadd(embedding_matrix)
x1.load_weights(f_path_noadditional)

loss, accuracy = x1.evaluate(test_X_pad_60len, label_test)
print('Accuracy:',(accuracy*100),'%')

x4= model_lstm_pooling_strlenjuk(embedding_matrix) 
x4.load_weights(f_path_all)

loss, accuracy =x4.evaluate([test_X_pad_60len, test_strlen, test_juk], label_test)
print('Accuracy:',(accuracy*100),'%')

def model_perfomance_noadd(model):
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
  print("ACCURACY (USING RULE): ",round(((TP+TN)/(P+N))*100,1),"%")
  e1 =(TP/P)*100
  e2 =(TP/(TP+FP))*100
  print("RECALL : ",round(e1,1),"%")
  print("PRECISION : ",round(e2,1),"%")
  print("F1 measure: ", round((2*e1*e2)/(e1+e2),1),"%" )
  print("F1 measure: ", round(2*((e1*e2)/(e1+e2)),1),"%" )
  df1 = pd.DataFrame([[TP,FP],[FN,TN]], range(2),range(2))
  ax= plt.subplot()
  sn.heatmap(df1, annot=True, ax = ax,fmt='d')       
  ax.set_xlabel('True labels')
  ax.set_ylabel('Predicted labels')
  ax.set_title('Confusion Matrix')
  ax.xaxis.set_ticklabels([ '1','0'])
  ax.yaxis.set_ticklabels(['1','0'])

def model_perfomance_all(model):
  test_result2 = model.predict([test_X_pad_60len,test_strlen, test_juk])
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
  print("ACCURACY (USING RULE): ",round(((TP+TN)/(P+N))*100,1),"%")
  e1 =(TP/P)*100
  e2 =(TP/(TP+FP))*100
  print("RECALL : ",round(e1,1),"%")
  print("PRECISION : ",round(e2,1),"%")
  print("F1 measure: ", round((2*e1*e2)/(e1+e2),1),"%" )
  print("F1 measure: ", round(2*((e1*e2)/(e1+e2)),1),"%" )
  df1 = pd.DataFrame([[TP,FP],[FN,TN]], range(2),range(2))
  ax= plt.subplot()
  sn.heatmap(df1, annot=True, ax = ax,fmt='d')       
  ax.set_xlabel('True labels')
  ax.set_ylabel('Predicted labels')
  ax.set_title('Confusion Matrix')
  ax.xaxis.set_ticklabels([ '1','0'])
  ax.yaxis.set_ticklabels(['1','0'])

print(model_perfomance_noadd(x1))

print(model_perfomance_all(x4))