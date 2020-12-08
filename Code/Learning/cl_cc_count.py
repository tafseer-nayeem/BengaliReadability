# -*- coding: utf-8 -*-


from sklearn.externals import joblib
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

trainSet = pd.read_csv('/content/drive/My Drive/Complete_dataset/models1/train.csv')
testSet = pd.read_csv('/content/drive/My Drive/Complete_dataset/models1/test.csv')
validSet = pd.read_csv('/content/drive/My Drive/Complete_dataset/models1/valid.csv')

sentence_train = list(trainSet['sentence'])
sentence_test = list(testSet['sentence'])
sentence_valid = list(validSet['sentence'])

all_sents = sentence_train + sentence_test + sentence_valid

all_words11 =[]

for i in range(len(all_sents)):
  v = all_sents[i].split(" ")
  for k in range(len(v)):
    all_words11.append(v[k])

all_words = list(set(all_words11))

def hh(inp):
  v =list(inp)
   
  for i in range(len(v)):
    if  i+1<len(v) and v[i] == 'া' and v[i+1] == 'ে':
     v[i] = 'ো'
     v[i+1] ='x'
  return v

def hh1(inp):
  v =list(inp)
   
  for i in range(len(v)):
    if  i-1>=0 and v[i] == '়':
      if v[i-1] =='য':
        v[i-1] = 'য়'
      elif v[i-1] =='ড':
        v[i-1] = 'ড়'
      elif v[i-1] =='ঢ':
        v[i-1] = 'ঢ়'
      elif v[i-1] =='ব':
        v[i-1] = 'র'
      v[i] = 'x'

  return v

input_words_list1 =[]
for i in range(len(all_words)):
  str1 = all_words[i]
  x =hh(hh1(str1))
  x1 =[]
  
  for i1 in range(len(x)):
   if x[i1]!='x':
      x1.append(x[i1])
  input_words_list1.append(''.join(x1))

all_words1 = input_words_list1

consonant_list =['ক','খ','গ', 'ঘ','ঙ', 'চ','ছ', 'জ', 'ঝ','ঞ','ট' ,'ঠ' ,'ড','ঢ','ণ' ,'ত' ,'থ' ,'দ' ,'ধ' ,'ন','প','ফ', 'ব','ভ', 'ম','য','র','ল' ,'শ','ষ' ,'স' ,'হ' ,'ৎ','ড়','ঢ়','য়']

virama_hasanta ='\u09CD'

all_words_juk = []

for i in range(len(all_words1)):
  wr = all_words1[i]

  wr24 = list(wr)
  for j in range(len(wr24)):
    if wr24[j] ==  '\u200c':
       wr24[j] = 'x'
  
  wr25 =[]
  
  for v in range(len(wr24)):
    if wr24[v] != 'x':
      wr25.append(wr24[v]) 

  wr23 = wr25
  
  if len(wr23)>=4:

    if wr23[1] ==virama_hasanta and wr23[2] =='য' and wr23[3] =='া':
       wr23[1] = 'x'

  wr11 = wr23
  
  for i1 in range(len(wr11)):
    if wr11[i1] == 'ৎ' :
      if i1-2>=0 and wr11[i1-1] == virama_hasanta and wr11[i1-2] == 'র': 

          pass
      else:
        wr11[i1] = 'ত্'

  
  
  wr1 = wr11

  wr2 =[]
  for k in range(len(wr1)):
    if wr1[k]!='x':
      wr2.append(wr1[k])

  word = ''.join(wr2)
  f = word
  #print(word, list(word))
  countt =0
  for k1 in range(len(f)):
    if f[k1] == virama_hasanta:

      if (k1-1)>=0 and (k1+1)<len(f): 
        if (k1-2)>=0:
           if f[k1-1] in consonant_list and f[k1+1] in consonant_list and f[k1-2]!= virama_hasanta:
             countt = countt +1
        else:
           if f[k1-1] in consonant_list and f[k1+1] in consonant_list:
             countt = countt +1
             
  all_words_juk.append(countt)

sentence_train[0]

train_juk =[]


for i in tqdm(range(len(sentence_train))):
  v =[]
  x = sentence_train[i].split(" ")
  x1 =[]
  for k in range(len(x)):
    m =hh(hh1(x[k]))
    temp = []
    for k1 in range(len(m)):
      
      if m[k1]!='x':
        temp.append(m[k1])
    x1.append(''.join(temp))
  
  for i1 in range(len(x1)):
      v.append(all_words_juk[all_words1.index(x1[i1])])
      
  train_juk.append(sum(v))

i= 13333
print(train_juk[i],"---->", sentence_train[i])

valid_juk =[]

for i in tqdm(range(len(sentence_valid))):
  v =[]
  x = sentence_valid[i].split(" ")
  x1 =[]
  for k in range(len(x)):
    m =hh(hh1(x[k]))
    temp = []
    for k1 in range(len(m)):
      
      if m[k1]!='x':
        temp.append(m[k1])
    x1.append(''.join(temp))
  
  for i1 in range(len(x1)):
      v.append(all_words_juk[all_words1.index(x1[i1])])
      
  valid_juk.append(sum(v))

i= 1163
print(valid_juk[i],"---->", sentence_valid[i])

test_juk =[]

for i in tqdm(range(len(sentence_test))):
  v =[]
  x = sentence_test[i].split(" ")
  x1 =[]
  for k in range(len(x)):
    m =hh(hh1(x[k]))
    temp = []
    for k1 in range(len(m)):
      
      if m[k1]!='x':
        temp.append(m[k1])
    x1.append(''.join(temp))
  
  for i1 in range(len(x1)):
      v.append(all_words_juk[all_words1.index(x1[i1])])
      
  test_juk.append(sum(v))

i= 2132
print(test_juk[i],"---->", sentence_test[i])

strlen_train =[]


for i in range(len(sentence_train)):
  strlen_train.append(len(sentence_train[i]))

strlen_test =[]


for i in range(len(sentence_test)):
  
  
  strlen_test.append(len(sentence_test[i]))

strlen_valid =[]


for i in range(len(sentence_valid)):

  
 strlen_valid.append(len(sentence_valid[i]))

train_juk1 =[]

for i in range(len(train_juk)):
  train_juk1.append(train_juk[i]+1)

test_juk1 =[]

for i in range(len(test_juk)):
  test_juk1.append(test_juk[i]+1)

valid_juk1 =[]

for i in range(len(valid_juk)):
  valid_juk1.append(valid_juk[i]+1)

fp1 = "/content/drive/My Drive/Complete_dataset/models1/train_juk.pkl"

joblib.dump(train_juk1, fp1)

fp2 ="/content/drive/My Drive/Complete_dataset/models1/test_juk.pkl"

joblib.dump(test_juk1, fp2)

fp3 ="/content/drive/My Drive/Complete_dataset/models1/valid_juk.pkl"

joblib.dump(valid_juk1, fp3)

fp4 ="/content/drive/My Drive/Complete_dataset/models1/strlen_train.pkl"

joblib.dump(strlen_train, fp4)

fp5 ="/content/drive/My Drive/Complete_dataset/models1/strlen_test.pkl"

joblib.dump(strlen_test, fp5)

fp6 ="/content/drive/My Drive/Complete_dataset/models1/strlen_valid.pkl"

joblib.dump(strlen_valid, fp6)



"""[https://bn.wikibooks.org/wiki/%E0%A6%AC%E0%A6%BE%E0%A6%82%E0%A6%B2%E0%A6%BE_%E0%A6%AF%E0%A7%81%E0%A6%95%E0%A7%8D%E0%A6%A4%E0%A6%BE%E0%A6%95%E0%A7%8D%E0%A6%B7%E0%A6%B0](https://bn.wikibooks.org/wiki/%E0%A6%AC%E0%A6%BE%E0%A6%82%E0%A6%B2%E0%A6%BE_%E0%A6%AF%E0%A7%81%E0%A6%95%E0%A7%8D%E0%A6%A4%E0%A6%BE%E0%A6%95%E0%A7%8D%E0%A6%B7%E0%A6%B0)"""

