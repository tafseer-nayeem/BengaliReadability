# -*- coding: utf-8 -*-

# goal 1: calculate cosine sim
# goal 2: Human annotation

 
import pandas as pd
import re
import numpy as np

from sklearn.externals import joblib
from scipy import spatial


from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

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

def bengali_sentence_tokenizer(text_path):
    file1 = open(text_path,"r")
    text1 = file1.read()
    l = sentenceSplit(singleSpace(preProcessing(singleDari(text1))))
    l1 =[]
    for i in range(len(l)):
            if (l[i].isspace()==False  and l[i]!=''): 
                l1.append(l[i])
    return l1


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
     
def cos(x1, x2):
  return np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))

numberofSimpleFile = 46
numberofComplexFile = 45

simple_temp =[]


for i in range(1, numberofSimpleFile+1):
  simple_temp.append(bengali_sentence_tokenizer("/content/drive/My Drive/Complete_dataset/simple/"+str(i)+".txt"))

simpleSentences =[]

for i in range(len(simple_temp)):
  for i1 in range(len(simple_temp[i])):
    simpleSentences.append(simple_temp[i][i1])

complex_temp =[]


for i in range(1, numberofComplexFile+1):
  complex_temp.append(bengali_sentence_tokenizer("/content/drive/My Drive/Complete_dataset/complex/"+str(i)+".txt"))

complexSentences =[]

for i in range(len(complex_temp)):
  for i1 in range(len(complex_temp[i])):
    complexSentences.append(complex_temp[i][i1])

simpleSentences1 = removeExtraWhitespaceFromSents(simpleSentences)
complexSentences1 = removeExtraWhitespaceFromSents(complexSentences)

"""[https://fasttext.cc/docs/en/crawl-vectors.html](https://fasttext.cc/docs/en/crawl-vectors.html)"""

#!wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.bin.gz

#!gunzip cc.bn.300.bin.gz

#!pip install fasttext

import fasttext

modelPy = fasttext.load_model("cc.bn.300.bin")

simpleSentences1_vector =[]
complexSentences1_vector =[]

for i in range(len(simpleSentences1)):
    simpleSentences1_vector.append(modelPy.get_sentence_vector(simpleSentences1[i]))

for i in range(len(complexSentences1)):
    complexSentences1_vector.append(modelPy.get_sentence_vector(complexSentences1[i]))

removeIndex = []

for i in tqdm(range(len(simpleSentences1_vector))):
  for i1 in range(len(complexSentences1_vector)):
    r = cos(simpleSentences1_vector[i], complexSentences1_vector[i1])
    if r >= 0.90:
      removeIndex.append(i1)

simpleWord =["হ্যাঁ","হাঁ","হ্যা","আচ্ছা","হুম","হুঁম","হা","না", "জ্বি","জ্বি না", "জ্বী", "জ্বী না"]

simpleWord2 =["করবো", "করব","খাবো", "ঘুমাব", " ঘুমাবো","পড়বো", "পড়ব","হবে", "হবেনা", "হবে না","নিবো", "দিবো"]

removeIndex = []
for i in tqdm(range(len(simpleWord2))):
  for i1 in range(len(complexSentences1_vector)):
    r = cos(modelPy.get_sentence_vector(simpleWord2[i]), complexSentences1_vector[i1])
    if r >= 0.95:
      removeIndex.append(i1)

for i in range(len(removeIndex)):
  print(removeIndex[i],"---",complexSentences1[removeIndex[i]])

simpleSentences1_vector_temp1 = simpleSentences1_vector[33000:len(simpleSentences1_vector)]

removeIndex2 = []

#!pip install tqdm

for i in tqdm(range(len(simpleSentences1_vector_temp1))):
  for i1 in range(len(complexSentences1_vector)):
    r = cos(simpleSentences1_vector_temp1[i], complexSentences1_vector[i1])
    if r >= 0.90:
      removeIndex2.append(i1)

removeIndex21 = list(set(removeIndex2))

for i in range(len(removeIndex21)):
  print(removeIndex21[i],"---",complexSentences1[removeIndex21[i]])

f= open("33000toRest.txt","w+")

for i in range(len(removeIndex21)):
  f.write(str(removeIndex21[i])+"\n")
f.close()

f1= open("xtraWords.txt","w+")
for i in range(len(removeIndex)):
  f1.write(str(removeIndex[i])+"\n")
f1.close()

f2 = open("/content/rmvIndex.txt","r")

lines = f2.readlines()

remvIndex =[]

for i in range(len(lines)):
  e = (lines[i]).replace("\n","")
  remvIndex.append(int(e))

remvIndex2 =list(set(remvIndex))

ff= open("index_v1.txt","w+")
for i in range(len(remvIndex2)):
  ff.write(str(remvIndex2[i])+"\n")
ff.close()

for i in range(2000,3186):
  print(i+1,"--",remvIndex2[i]," = ",complexSentences1[remvIndex2[i]])

simpleInComplex_human =[24595,40980 ,8263, 16565,8307, 57564, 57570,49401,244,25046,49628, 25052, 8686,49648, 8701, 16915, 49699, 8849, 8957 ,33551 ,910 ,50099 ,17348 ,58430 ,50352 ,50355,58601 ,42397,42455,34366 ,9805,9865, 9876 , 42694 ,10045 ,18236, 42818,59255,51072 , 59269, 26505 , 1932 ,10142 ,26537 ,10188,2011 , 51237, 59432 , 10286, 59452 ,51263, 51273 ,43092 , 43094,10444 , 59606, 2271 ,43242 , 10480 , 43281 , 43292 , 35105,10630 , 18846 ,18923, 51710 , 59923 ,60031 ,11017, 11019 , 11057 , 35638 , 11068,27506 , 11170, 35755 , 35803, 35837 , 52288, 52314 , 11356 , 52338, 60540, 52397, 60611, 52438, 11534, 52524, 3380 ,11582, 3403, 60765 , 60788 , 36225 , 60820 , 60821, 60830, 60834, 36310, 36311, 11737,3619 , 20004 ,20038, 11923, 11976, 52938, 12074 , 44904 , 12193, 53166 , 53290 , 53294, 20627, 37012, 28891, 4348, 4358 , 45339, 12669, 12732 ,12730, 12736, 45527 , 29146, 37343, 12796, 53843, 12929, 4773,4880 , 54032, 13101 , 29484, 37681 , 13109, 21308, 54103, 13165, 4976 , 29588  , 29624, 54277, 46090 , 29730, 13389 , 54577, 13660 ,  13665,  5558, 22034, 22036, 22075, 5810,  46794 , 14050, 14051 ,  30449, 55050, 55062, 14195, 6022, 14344, 6234,  6314 , 22722, 39124, 30936, 22803, 6421, 22810, 14713, 31192, 6630, 39401, 23054, 6921, 47985, 15279 ,23598,40002,  40035,  31883, 23915 , 15745, 7609,7610, 32271,24166 ,24174, 48755, 40749, 8001, 32592, 16251, 8091, 8105 , 57264,  32687 ,8159, 8161, 8165, 8187]

"""simpleInComplex_human =[24595,40980 ,8263, 16565,8307, 57564, 57570,49401,244,25046,49628, 25052, 8686,49648, 8701, 16915, 49699, 8849, 8957 ,33551 ,910 ,50099 ,17348 ,58430 ,50352 ,50355,58601 ,42397,42455,34366 ,9805,9865, 9876 , 42694 ,10045 ,18236, 42818,59255,51072 , 59269, 26505 , 1932 ,10142 ,26537 ,10188,2011 , 51237, 59432 , 10286, 59452 ,51263, 51273 ,43092 , 43094,10444 , 59606, 2271 ,43242 , 10480 , 43281 , 43292 , 35105,10630 , 18846 ,18923, 51710 , 59923 ,60031 ,11017, 11019 , 11057 , 35638 , 11068,27506 , 11170, 35755 , 35803, 35837 , 52288, 52314 , 11356 , 52338, 60540, 52397, 60611, 52438, 11534, 52524, 3380 ,11582, 3403, 60765 , 60788 , 36225 , 60820 , 60821, 60830, 60834, 36310, 36311, 11737,3619 , 20004 ,20038, 11923, 11976, 52938, 12074 , 44904 , 12193, 53166 , 53290 , 53294, 20627, 37012, 28891, 4348, 4358 , 45339, 12669, 12732 ,12730, 12736, 45527 , 29146, 37343, 12796, 53843, 12929, 4773,4880 , 54032, 13101 , 29484, 37681 , 13109, 21308, 54103, 13165, 4976 , 29588  , 29624, 54277, 46090 , 29730, 13389 , 54577, 13660 ,  13665,  5558, 22034, 22036, 22075, 5810,  46794 , 14050, 14051 ,  30449, 55050, 55062, 14195, 6022, 14344, 6234,  6314 , 22722, 39124, 30936, 22803, 6421, 22810, 14713, 31192, 6630, 39401, 23054, 6921, 47985, 15279 ,23598,40002,  40035,  31883, 23915 , 15745, 7609,7610, 32271,24166 ,24174, 48755, 40749, 8001, 32592, 16251, 8091, 8105 , 57264,  32687 ,8159, 8161, 8165, 8187]"""

ff= open("/content/index_v1.txt","r")

x = ff.readlines()

rmvIndx =[]

for i in range(len(x)):
  r = int(x[i].replace("\n",""))
  if r not in simpleInComplex_human:
     rmvIndx.append(r)

complexSentences11 =[]



for i in tqdm(range(len(complexSentences1))):
    if i  in rmvIndx:
      complexSentences11.append(complexSentences1[i])

filenamex ="simpleSentences.pkl"
joblib.dump(simpleSentences1,filenamex)

filenamex1 ="complexSentences.pkl"
joblib.dump(complexSentences13,filenamex1)

cv = joblib.load("/content/drive/My Drive/Complete_dataset/models/complexSentences.pkl")

cv1 =cv

remvAgain =[]

for i in range(len(simpleWord2)):
 for i1 in range(len(cv1)):
       c = cos(modelPy.get_sentence_vector(simpleWord2[i]),modelPy.get_sentence_vector(cv1[i1]))
       if c>=0.95:
         remvAgain.append(i1)

temp =[]
for i in range(len(remvAgain)):
  print(remvAgain[i],"---",cv1[remvAgain[i]])
  temp.append(cv1[remvAgain[i]])

complexSentences13 =[]

for i in range(len(cv1)):
  if cv1[i] not in temp:
      complexSentences13.append(cv1[i])

simp = joblib.load("/content/drive/My Drive/Complete_dataset/models1/simpleSentences.pkl")
comp = joblib.load("/content/drive/My Drive/Complete_dataset/models1/complexSentences.pkl")

simp_x =[]


for i in range(len(simp)):
   x = simp[i].replace("…",'')
   if len(x)>0:
     simp_x.append(x)

comp_x =[]


for i in range(len(comp)):
   x = comp[i].replace("…",'')
   if len(x)>0:
     comp_x.append(x)

filenamex ="simpleSentences.pkl"
joblib.dump(simp_x,filenamex)

filenamex1 ="complexSentences.pkl"
joblib.dump(comp_x,filenamex1)