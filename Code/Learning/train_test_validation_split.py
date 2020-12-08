# -*- coding: utf-8 -*-

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from scipy import spatial
import random
import numpy as np
import seaborn as sn
from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def removeExtraWhitespaceFromSents(inp):
   sentence_token =[]
   for i in range(len(inp)):
     k =[]
     removed_index =[]
     k = inp[i].split(" ")
     sentence_token.append(list(filter(str.strip, k)))
   sentence_token_merged =[]
   for i in range(len(sentence_token)):
     sentence_token_merged.append(' '.join(sentence_token[i]))
   return sentence_token_merged

simpleSentences_temp = joblib.load("/content/drive/My Drive/Complete_dataset/models1/simpleSentences.pkl")
complexSentences_temp = joblib.load("/content/drive/My Drive/Complete_dataset/models1/complexSentences.pkl")

simpleSentences2 = removeExtraWhitespaceFromSents(simpleSentences_temp)
complexSentences2 = removeExtraWhitespaceFromSents(complexSentences_temp)

simpleSentences1 = list(set(simpleSentences2))
complexSentences1 = list(set(complexSentences2))

#len(simpleSentences1 + complexSentences1)

#len(simpleSentences1)

#len(complexSentences1)

rmv =[]

for i in tqdm(range(len(complexSentences1))):
  if complexSentences1[i] in simpleSentences1:
    rmv.append(complexSentences1[i])

#rmv

rmv = list(set(rmv))

#rmv

complexSentences= []


for i in range(len(complexSentences1)):
  if complexSentences1[i] not in rmv:
    complexSentences.append(complexSentences1[i])

simpleSentences = simpleSentences1

filenamex1 ="/content/drive/My Drive/Complete_dataset/models1/simpleSentences_updated.pkl"
joblib.dump(simpleSentences,filenamex1)

filenamex2 ="/content/drive/My Drive/Complete_dataset/models1/complexSentences_updated.pkl"
joblib.dump(complexSentences,filenamex2)

simp = simpleSentences
comp = complexSentences

allIndx_simp1 = [i for i in range(len(simp))]
allIndx_comp1 = [i for i in range(len(comp))]

allIndx_simp = shuffle(allIndx_simp1)
allIndx_comp = shuffle(allIndx_comp1)

simp_indx = random.sample(allIndx_simp, 2200)

comp_indx = random.sample(allIndx_comp, 2200)

simp_indx_test =random.sample(simp_indx, 1100)

simp_indx_valid = []

for i in range(len(simp_indx)):
  if simp_indx[i] not in simp_indx_test:
    simp_indx_valid.append(simp_indx[i])

test_valid1 = simp_indx_test + simp_indx_valid

#len(set(test_valid1))

simp_indx_train = []

for i in range(len(allIndx_simp)):
  if allIndx_simp[i] not in test_valid1:
     simp_indx_train.append(allIndx_simp[i])

comp_indx_test =random.sample(comp_indx, 1100)

comp_indx_valid = []

for i in range(len(comp_indx)):
  if comp_indx[i] not in comp_indx_test:
   comp_indx_valid.append(comp_indx[i])

test_valid2 = comp_indx_test + comp_indx_valid

#len(set(test_valid2))

comp_indx_train = []

for i in range(len(allIndx_comp)):
  if allIndx_comp[i] not in test_valid2:
     comp_indx_train.append(allIndx_comp[i])

#len(comp_indx_test + comp_indx_train + comp_indx_valid)

#len(simp_indx_test + simp_indx_train + simp_indx_valid)

"""**Training Set**"""

sentence_train1 = []

for i in range(len(simp_indx_train)):
  sentence_train1.append(simp[simp_indx_train[i]])

for i in range(len(comp_indx_train)): 
  sentence_train1.append(comp[comp_indx_train[i]])

label_train1 = list(np.ones(len(simp_indx_train), dtype=int)) + list(np.zeros(len(comp_indx_train), dtype=int))

sentence_train,label_train = shuffle(sentence_train1,label_train1)

trainset = pd.DataFrame({
    'sentence': sentence_train,
     'class': label_train 
     
    })

trainset.to_csv('/content/drive/My Drive/Complete_dataset/models1/train.csv',index=False)



"""**Validation Set**"""

sentence_valid1 = []

for i in range(len(simp_indx_valid)):
  sentence_valid1.append(simp[simp_indx_valid[i]])

for i in range(len(comp_indx_valid)):
  sentence_valid1.append(comp[comp_indx_valid[i]])

label_valid1 = list(np.ones(len(simp_indx_valid), dtype=int)) + list(np.zeros(len(comp_indx_valid), dtype=int))

sentence_valid,label_valid = shuffle(sentence_valid1,label_valid1)

validset = pd.DataFrame({
    'sentence': sentence_valid,
     'class': label_valid 
     
    })

validset.to_csv('/content/drive/My Drive/Complete_dataset/models1/valid.csv',index=False)

"""**Test Set**"""

sentence_test1 = []

for i in range(len(simp_indx_test)):
  sentence_test1.append(simp[simp_indx_test[i]])

for i in range(len(comp_indx_test)):
  sentence_test1.append(comp[comp_indx_test[i]])

label_test1 = list(np.ones(len(simp_indx_test), dtype=int)) + list(np.zeros(len(comp_indx_test), dtype=int))

sentence_test,label_test = shuffle(sentence_test1,label_test1)

testset = pd.DataFrame({
    'sentence': sentence_test,
     'class': label_test 
     
    })

testset.to_csv('/content/drive/My Drive/Complete_dataset/models1/test.csv',index=False)

