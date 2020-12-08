# -*- coding: utf-8 -*-

import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from numpy import array
from gensim.models import Word2Vec

from sklearn.externals import joblib
from scipy import spatial


import seaborn as sn
import matplotlib.pyplot as plt

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#!pip install fasttext

import fasttext

trainSet = pd.read_csv('/content/drive/My Drive/Complete_dataset/models1/train.csv')

testSet = pd.read_csv('/content/drive/My Drive/Complete_dataset/models1/test.csv')

print(len(list(trainSet['sentence'])))

sentence_train = list(trainSet['sentence'])
sentence_test = list(testSet['sentence'])
label_train = list(trainSet['class'])
label_test  = list(testSet['class'])

label_train2 =[]



for i in range(len(label_train)):
  s = "__label__"+str(label_train[i])
  label_train2.append(s)

fastText_dataset =[]


for i in tqdm(range(len(sentence_train))):
   c = label_train2[i]+" "+sentence_train[i]
   fastText_dataset.append(c)

#len(fastText_dataset)

f= open("/content/drive/My Drive/Complete_dataset/models1/fastText_train_dataset.txt","w+")

for i in range(len(fastText_dataset)):
  if i == len(fastText_dataset)-1:
    f.write(fastText_dataset[i])
  else:
    f.write(fastText_dataset[i]+"\n")


f.close()

model_fasttext_unigram = fasttext.train_supervised('/content/drive/My Drive/Complete_dataset/models1/fastText_train_dataset.txt',ws =2, minn=2,maxn =6, lr=0.5, epoch=50, loss='hs')

model_fasttext_bigram = fasttext.train_supervised('/content/drive/My Drive/Complete_dataset/models1/fastText_train_dataset.txt',ws =2,minn=2,maxn =6, lr=0.5, epoch=50, loss='hs',wordNgrams=2)

model_fasttext_trigram = fasttext.train_supervised('/content/drive/My Drive/Complete_dataset/models1/fastText_train_dataset.txt',ws =2, minn=2,maxn =6, lr=0.5, epoch=50, loss='hs', wordNgrams=3 )

"""**MODEL PERFORMANCE**"""

def model_performance(model):
   predict_label =[]


   for i in range(len(sentence_test)):
      c = list(model.predict(sentence_test[i]))[0][0]
      predict_label.append(int(c[len(c)-1]))
   
   correct =0
   for i in range(len(label_test)):
       if predict_label[i] == label_test[i]:
          correct = correct +1
   print(correct,"out of ",len(label_test),"correctly classified!" )
   print("Accuracy: ", (correct/len(label_test))*100,"%")
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

#model_performance(model_fasttext_unigram)

#model_performance(model_fasttext_bigram)

#model_performance(model_fasttext_trigram)

model_fasttext_unigram.save_model("/content/drive/My Drive/Complete_dataset/models1/readability_fastext_model_uni.bin")

model_fasttext_bigram.save_model("/content/drive/My Drive/Complete_dataset/models1/readability_fastext_model_bi.bin")

model_fasttext_trigram.save_model("/content/drive/My Drive/Complete_dataset/models1/readability_fastext_model_tri.bin")

