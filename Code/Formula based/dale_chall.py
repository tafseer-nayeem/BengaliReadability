# -*- coding: utf-8 -*-

import re
import pandas as pd
from sklearn.externals import joblib
from tqdm import tqdm
import numpy as np
import math

"""**Stemmer source:**  

[https://github.com/MIProtick/Bangla-stemmer](https://github.com/MIProtick/Bangla-stemmer)
"""

#!pip install bangla_stemmer

from bangla_stemmer.stemmer import stemmer

f_path1 = "/content/drive/My Drive/Complete_dataset/models1/dale_chall_3396words.pkl"
dale_chall_3396words = joblib.load(f_path1)

def hh(inp):
  v =list(inp)
   
  for i in range(len(v)):
    if i+1<len(v) and v[i] == 'া' and v[i+1] == 'ে':
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

stmr = stemmer.BanglaStemmer()

dale_chall_3396words_stemmed =[]

for i in tqdm(range(len(dale_chall_3396words))):
  dale_chall_3396words_stemmed.append(stmr.stem(dale_chall_3396words[i]))

for i in range(len(dale_chall_3396words)):
  print(i, dale_chall_3396words[i], dale_chall_3396words_stemmed[i])

def difficult_words_count(input_words_list, easy_words_list):
 diff =0
 
 input_words_list1 =[]
 for i in range(len(input_words_list)):
  str1 = input_words_list[i]
  x =hh(hh1(str1))
  x1 =[]
  
  for i1 in range(len(x)):
   if x[i1]!='x':
      x1.append(x[i1])
  input_words_list1.append(''.join(x1)) 
  
 input_words_list11 =[]
 for g in range(len(input_words_list1)):
   input_words_list11.append(stmr.stem(input_words_list1[g]))

 for i in range(len(input_words_list11)):
   if input_words_list11[i] not in easy_words_list:
     #print(input_words_list1[i]) 
     diff = diff +1

 return diff

def singleDari(s1):
    s2 = re.sub(r'\n+', '।',s1)
    s3 = re.sub(r'।+\s*।*', '।',s2)
    return s3

def singleSpace(ss):
  return re.sub(r'\s+\s*',' ',ss)


def sentenceSplit(str1):
   return re.split(r'।|\?|!',str1)



def replaceMultiple(mainString, toBeReplaces, newString): 


    for elem in toBeReplaces:
        if elem in mainString:
            mainString = mainString.replace(elem, newString)

    return mainString



def preProcessing(s):
    s1 = replaceMultiple(s, ["…","\\",'{','}','[',']','॥','#','”','“','.',',',';',':','/','"','–','-','*','(',')','\'','%','$', '&', '+', '=','<', '>','|','—','_','\ufeff','\u200c','’','‘'] ,' ')
    return s1

def bengali_sentence_tokenizer(text):
    l = sentenceSplit(singleSpace(preProcessing(singleDari(text))))
    l1 =[]
    for i in range(len(l)):
            if (l[i].isspace()==False  and l[i]!=''): 
                l1.append(l[i])
    return l1

def dale_chall(text_path):
  file1 = open(text_path,"r")
  text = file1.read()

  if len(text.strip()) != 0: 
    s1= bengali_sentence_tokenizer(text)      
    sentence_token =[]

    for i in range(len(s1)):
       k =[]
  
       k = s1[i].split(" ")

       sentence_token.append(list(filter(str.strip, k)))
    words = 0
    input_length_list = []
    for i in range(len(sentence_token)):
      input_length_list.append(len(sentence_token[i]))
    words =sum(input_length_list)
    sentences = len(s1)

    all_words_list =[]
    for i in range(len(sentence_token)):
      for j in range(len(sentence_token[i])):
        all_words_list.append(sentence_token[i][j])

    print(words)
    print(sentences)

    words_per_sent = words/sentences
    diff_words = difficult_words_count(all_words_list, dale_chall_3396words_stemmed)
    diff_words_percentage = (diff_words/words) * 100
    print("Difficult words percentage:", diff_words_percentage)
    raw_score = 0.1579 * diff_words_percentage + 0.0496 * words_per_sent
    adjusted_score = raw_score + 3.6365
    
    if diff_words_percentage > 5:
      return round(adjusted_score,1)
    else:
      return round(raw_score,1) 
  else:
    return "Your input is empty!!!!!!!!!!"

print(dale_chall('/content/drive/My Drive/Readability_dataset/ARI/class4_1.txt'))

