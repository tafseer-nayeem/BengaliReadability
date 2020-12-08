# -*- coding: utf-8 -*-

import re
import pandas as pd
import joblib
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

def bengali_sentence_tokenizer(text_path):
    file1 = open(text_path,"r")
    text = file1.read()
    l = sentenceSplit(singleSpace(preProcessing(singleDari(text))))
    l1 =[]
    for i in range(len(l)):
            if (l[i].isspace()==False  and l[i]!=''): 
                l1.append(l[i])
    return l1

def bengali_sentence_tokenizer1(text):
    l = sentenceSplit(singleSpace(preProcessing(singleDari(text))))
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

pronunciation_dict = pd.read_csv("/content/drive/My Drive/Readability_dataset/lexicon.tsv",sep='\t',header=None)

pronunciation_dict_words=list(pronunciation_dict[0])

pronunciation_dict_length_temp = list(pronunciation_dict[1])

pronunciation_dict_words_length = []



for i in range(len(pronunciation_dict_length_temp)):
    str1 = pronunciation_dict_length_temp[i]
    if '.' not in str1:
      pronunciation_dict_words_length.append(1)
    else:
      pronunciation_dict_words_length.append(len(str1.split('.')))

g1 = '/content/drive/My Drive/Readability_dataset/pronunciation_dict_words.pkl'
joblib.dump(pronunciation_dict_words,g1)

g2 ='/content/drive/My Drive/Readability_dataset/pronunciation_dict_words_length.pkl'
joblib.dump(pronunciation_dict_words_length,g2)

s1= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/class1_1.txt')


s5= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/class2_1.txt')


s9= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/class3_1.txt')

s13= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/class4_1.txt')


s17= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/class5_1.txt')


s21= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/class6_1.txt')


s25= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/class7_1.txt')


s29= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/class8_1.txt')



s33= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/class9+10_1.txt')


s37= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/class11+12_1.txt')



s41= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/shishutosh1.txt')
s42= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/shishutosh2.txt')



s46= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/prapto1.txt')
s47= bengali_sentence_tokenizer('/content/drive/My Drive/Readability_dataset/ARI/prapto2.txt')

all_sents =s1+s5+s9+s13+s17+s21+s25+s29+s33+s37+s41+s42+s46+s47

all_sents1 = removeExtraWhitespaceFromSents(all_sents)

words =[]


for i in range(len(all_sents1)):
    s = all_sents1[i].split()
    for i1 in range(len(s)):
      words.append(s[i1])

unique_words = list(set(words))

vocab_not_in_dict =[]

for i in range(len(unique_words)):
  if unique_words[i] not in pronunciation_dict_words:
    #count2 = count2 + 1
    vocab_not_in_dict.append( unique_words[i])

f=open("/content/drive/My Drive/Readability_dataset/vocab_not_in_dict.txt","w+")

for i in range(len(vocab_not_in_dict)):
  f.write(vocab_not_in_dict[i]+","+"\n")

oov1 = pd.read_csv("/content/drive/My Drive/Readability_dataset/vocab_not_in_dict.csv",header=None)

all_words =pronunciation_dict_words + list(oov1[0])

all_words_syllable_count =  pronunciation_dict_words_length +  list(oov1[1])

"""**Flesch–Kincaid readability tests**
[https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests)
"""

def flesch_kincaid(text_path):
   file1 = open(text_path,"r")
   text = file1.read()
   if len(text.strip()) != 0: 
     s1= bengali_sentence_tokenizer1(text)      
     sentence_token =[]
     for i in range(len(s1)):
        k =[]
        k = s1[i].split(" ")
        sentence_token.append(list(filter(str.strip, k)))

     input_length_list = []
     for i in range(len(sentence_token)):
       input_length_list.append(len(sentence_token[i]))
     
     total_words =sum(input_length_list)
     total_sentences = len(s1)
     
     my_words =[]
     syllable_count =[]
     
     for i in range(len(sentence_token)):
        for i1 in range(len(sentence_token[i])):
          my_words.append(sentence_token[i][i1]) 
     

     for i in range(len(my_words)):
       if my_words[i] == "দিল" or my_words[i] == "পাইল":
           syllable_count.append(2)
       
       else:    
        for i1 in range(len(all_words)):
         
          if my_words[i] == all_words[i1]:
            syllable_count.append(all_words_syllable_count[i1])
            break
     print("sents:", total_sentences)       
     print("words:",total_words) 
     print(len(my_words),"   ", len(syllable_count))
     
     total_syllables = sum(syllable_count)
     print(total_syllables)
     words_per_sent= total_words/total_sentences
     syllables_per_word = total_syllables/total_words
        
     flesch_score = 206.835 - (1.015 * words_per_sent) -(84.6 * syllables_per_word)
     print("flesch_score: ", round(flesch_score,2))
     
     kincaid = (0.39 * words_per_sent) + (11.8 * syllables_per_word) - 15.59
     print("Flesch–Kincaid grade level: ",round(kincaid,2))

"""**Gunning fog Index**


[https://www.webfx.com/tools/read-able/gunning-fog.html](https://www.webfx.com/tools/read-able/gunning-fog.html)
"""

#!pip install bnlp_toolkit

#!pip install fasttext

import nltk
nltk.download("punkt")

from bnlp.bengali_pos import BN_CRF_POS
bn_pos = BN_CRF_POS()
model_path = "/content/drive/My Drive/Readability_dataset/bn_pos_model.pkl"

def gunning_fog(text_path):
   file1 = open(text_path,"r")
   text = file1.read()
   if len(text.strip()) != 0: 
     s1= bengali_sentence_tokenizer1(text)      
     sentence_token =[]
     for i in range(len(s1)):
        k =[]
        k = s1[i].split(" ")
        sentence_token.append(list(filter(str.strip, k)))

     input_length_list = []
     for i in range(len(sentence_token)):
       input_length_list.append(len(sentence_token[i]))
     
     total_words =sum(input_length_list)
     total_sentences = len(s1)
     
     my_words =[]
     syllable_count =[]
     
     for i in range(len(sentence_token)):
        for i1 in range(len(sentence_token[i])):
          my_words.append(sentence_token[i][i1]) 
     

     for i in range(len(my_words)):
       if my_words[i] == "দিল" or my_words[i] == "পাইল":
           syllable_count.append(2)
       
       else:    
        for i1 in range(len(all_words)):
         
          if my_words[i] == all_words[i1]:
            syllable_count.append(all_words_syllable_count[i1])
            break
     
     complex_words = 0
     for i in range(len(s1)):
        #print(s1[i]) 
        res = bn_pos.pos_tag(model_path, s1[i])
        for i1 in range(len(res)):
          if res[i1][1]!='NP' and  all_words_syllable_count[all_words.index(res[i1][0])] >=3:
            complex_words = complex_words + 1
            #print( res[i1][1]," ", res[i1][0], " ", complex_words)
     
     
     words_per_sent= total_words/total_sentences
     complexWords_per_word = complex_words/total_words
     
     g = 0.4 * (words_per_sent + 100 * complexWords_per_word)
     
     print("Gunning fog index: ",round(g))

"""SMOG"""

def smog(text_path):
   file1 = open(text_path,"r")
   text = file1.read()
   if len(text.strip()) != 0: 
     s1= bengali_sentence_tokenizer1(text)      
     sentence_token =[]
     for i in range(len(s1)):
        k =[]
        k = s1[i].split(" ")
        sentence_token.append(list(filter(str.strip, k)))

     input_length_list = []
     for i in range(len(sentence_token)):
       input_length_list.append(len(sentence_token[i]))
     
     total_words =sum(input_length_list)
     total_sentences = len(s1)
     
     my_words =[]
     syllable_count =[]
     
     for i in range(len(sentence_token)):
        for i1 in range(len(sentence_token[i])):
          my_words.append(sentence_token[i][i1]) 
     

     for i in range(len(my_words)):
       if my_words[i] == "দিল" or my_words[i] == "পাইল":
           syllable_count.append(2)
       
       else:    
        for i1 in range(len(all_words)):
         
          if my_words[i] == all_words[i1]:
            syllable_count.append(all_words_syllable_count[i1])
            break
     
     number_of_polySyllables =0
     
     for i in range(len(syllable_count)):
       if syllable_count[i] >=3:
            number_of_polySyllables = number_of_polySyllables + 1
     

     s = 1.0430 * math.sqrt((30 * number_of_polySyllables) / total_sentences) + 3.1291
     print("SMOG score: ",round(s,2))