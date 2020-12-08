# -*- coding: utf-8 -*-
import re
import pandas as pd
import math

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



def ARI(text_path):
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
    characters_count_temp=[]
    for i in range(len(sentence_token)):
      for j in range(len(sentence_token[i])):
        #print(sentence_token[i][j],'----',len(list(sentence_token[i][j])))
        characters_count_temp.append(len(list(sentence_token[i][j])))  
    characters = sum(characters_count_temp)
    sentences = len(s1)
    print(characters)
    print(words)
    print(sentences)
    ARI_score = (4.71*(characters/ words)) + (0.5*(words/sentences))  - 21.43
    return math.ceil(ARI_score)
  else:
    return "Your input is empty!!!!!!!!!!"