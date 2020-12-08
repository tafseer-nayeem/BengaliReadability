# -*- coding: utf-8 -*-

f_path = "/content/consonant_conjunct_dataset.txt"

f=open(f_path, "r")

f1 = f.readlines()

all_words = []
all_labels = []

for i in range(len(f1)):
  v = f1[i].split(",")
  all_words.append(v[0])
  all_labels.append(int(v[1].replace('\n','').strip()))

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

actual_labels = all_labels
predict_labels = all_words_juk

correct_count = 0


for i in range(len(actual_labels)):
  if actual_labels[i] == predict_labels[i]:
    correct_count = correct_count + 1

print("Accuracy: ", (correct_count/len(actual_labels))*100,"%")

#100 percent accuracy found