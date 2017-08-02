from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import re
import os
import math
#from nltk.stem.porter import PorterStemmer
from nltk.stem import PorterStemmer

file1= dict()
count= 0    
wordcountlist=[]
dummycount=[]
i=0;
dictone = {}


###///////////////////// Traning data Preprocessing

print("PreProcessing training data")
for foldername in os.listdir("20_newsgroups_1000/"):
  test= ""
  filename= "20_newsgroups_1000/" + foldername

  #filename= "mini_newsgroups/" + foldername
  for text in os.listdir(filename)[:500]:
 #            print (foldername,text)
         with open(os.path.join(filename,text)) as file_read:
             test+= file_read.read()
         file_read.close()
  filtered_word = []
  tokenizer = RegexpTokenizer(r'\w+')
  #tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

  word = re.sub(r'\d+', '',test)
  toke=tokenizer.tokenize(word.lower())
  stopword = set(stopwords.words("english"))
  for w in toke:
          if w not in stopword:
              filtered_word.append(w)
  stemmer = PorterStemmer()
  filtered_words=[]
  filtered_words = [stemmer.stem(token) for token in filtered_word]
  tokens=Counter(filtered_words)
  for k,v in tokens.items():
      if k in dictone:
          pass
      else:
          dictone[k]=v    
  dummy = len(tokens)
  count=len(dictone)      
  wordcount=0
  #d=len(dummyunique)
  wordcount +=sum([tokens.get(token) for token in tokens])
  file1[foldername] = tokens
  #file1[foldername]['word_count'] = wordcount
  
  #file1[foldername]['uniquecount'] = dummy

  dummycount.append(dummy)
  wordcountlist.append(wordcount)
  i=i+1
  f=foldername

i=0
flag=0
xt=0
i=0
predict= {}
class_count=0
filtered_words1=[]

#///////////////////////////////Test Data 

print("PreProcessing test data")

for foldername2 in os.listdir("20_newsgroups_1000/"):
  filename1= "20_newsgroups_1000/" + foldername2
  text3=""
  for text1 in os.listdir(filename1)[500:]:
      with open(os.path.join(filename1,text1)) as text2:
          text3= text2.read()
      text2.close()
      filtered_word1 = []
      tokenizer1 = RegexpTokenizer(r'\w+')
      #tokenizer1= RegexpTokenizer(r'[a-zA-Z]+')

      word1 = re.sub(r'\d+', '',text3)
      tokens1=tokenizer.tokenize(word1.lower())
      stopword = set(stopwords.words("english"))
      for w in tokens1:
              if w not in stopword:
                  filtered_word1.append(w)
      stemmer = PorterStemmer()
      filtered_words1 = [stemmer.stem(token) for token in filtered_word1]

      
#/////////////////////////////Probability Calculation      

      probabilitydict={}

      flag=0
      for item in file1:
        probability=0
        
        for word in filtered_words1:
                probability = probability + math.log((file1[item][word]+1)/20*(wordcountlist[flag]+count+1)))

        probabilitydict[item]= probability
        #print(flag)
        flag+=1
      i+=1
      predict[text1] = max(probabilitydict, key= probabilitydict.get )
      #print (predict)
      if(foldername2==predict[text1]):
          class_count+=1

#////////////////////////////////////Accuracy    

accuracy = (class_count/i)*100
print ( "The accuracy of the Naive Bayes Classifier = ",accuracy)
    
