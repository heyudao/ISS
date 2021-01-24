# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 17:03:52 2019

@author: isswan
"""


import sklearn
import nltk
import re 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer #as vectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.linear_model import LogisticRegression

##########################################################################
##prepare dataset
from os import getcwd, chdir
import pandas as pd
import pickle as pk
fpath = getcwd()
print (fpath)
# Change your path here
chdir(fpath) 


training = pd.read_csv(fpath+"\\Data\\train.csv", encoding="utf-8")
testing  = pd.read_csv(fpath+"\\Data\\test.csv", encoding="utf-8")
print (training.head(5))
print (testing.head(5))
print (testing.tail(5))

#Label conversion: Positive to 1,Negative to -1 
train_pos = training[(training.Sentiment == 'positive')]
train_neg = training[(training.Sentiment == 'negative')]
print (train_pos.head(3))
print (train_pos.head(3))

train_pos_list = []
for i,t in train_pos.iterrows():
    train_pos_list.append([t.text.lower(), 1])

train_neg_list = []
for i,t in train_neg.iterrows():
    train_neg_list.append([t.text.lower(), -1])
#Same for test dataset   
test_pos = testing[(testing.Sentiment == 'positive')]
test_neg = testing[(testing.Sentiment == 'negative')]

test_pos_list = []
for i,t in test_pos.iterrows():
    test_pos_list.append([t.text.lower(), 1])

test_neg_list = []
for i,t in test_neg.iterrows():
    test_neg_list.append([t.text.lower(), -1])

#build the two dataset
trainset = train_pos_list + train_neg_list
testset = test_pos_list + test_neg_list

##########################################################3
###Preprocessing 
# seperate the text with labels

X_train = [t[0] for t in trainset]
X_test = [t[0] for t in testset]

Y_train = [t[1] for t in trainset]
Y_test = [t[1] for t in testset]

#Vectorizer the sentences using Tfidf vale
#Make sure test data should be transformed using vectorizer learned from trainning data 
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

# same feature set
train_vectors.shape
test_vectors.shape

#########################################################

#ADD Features - Negation
import re 
def nega_tag(text):
    transformed = re.sub(r"\b(?:never|nothing|nowhere|noone|none|not|haven't|hasn't|hasnt|hadn't|hadnt|can't|cant|couldn't|couldnt|shouldn't|shouldnt|won't|wont|wouldn't|wouldnt|don't|dont|doesn't|doesnt|didn't|didnt|isnt|isn't|aren't|arent|aint|ain't|hardly|seldom)\b[\w\s]+[^\w\s]", lambda match: re.sub(r'(\s+)(\w+)', r'\1NEG_\2', match.group(0)), text, flags=re.IGNORECASE)
    return(transformed)

text = "I don't like that place, you keep calling awesome."
print (nega_tag(text))

# Create a training list which will now contain reviews with Negatively tagged words and their labels
train_set_nega = []

# Append elements to the list
for doc in trainset:
    trans = nega_tag(doc[0])
    lab = doc[1]
    train_set_nega.append([trans, lab])

print(train_set_nega[18])

# Create a testing list which will now contain reviews with Negatively tagged words and their labels
test_set_nega = []

# Append elements to the list
for doc in testset:
    trans = nega_tag(doc[0])
    lab = doc[1]
    test_set_nega.append([trans, lab])


#Redo - Preprocessing 
# seperate the text with labels


X_nega_train = [t[0] for t in train_set_nega]
X_nega_test = [t[0] for t in test_set_nega]

Y_nega_train = [t[1] for t in train_set_nega]
Y_nega_test = [t[1] for t in test_set_nega]

#Vectorizer the sentences using Tfidf vale
#Make sure test data should be transformed using vectorizer learned from trainning data 
vectorizer = TfidfVectorizer()
train_nega_vectors = vectorizer.fit_transform(X_nega_train)
test_nega_vectors = vectorizer.transform(X_nega_test)

# bigger feature set
train_vectors.shape
test_vectors.shape

train_nega_vectors.shape
test_nega_vectors.shape

##########################################################################################
###Bigram
bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
train_bigram_vectors = bigram_vectorizer.fit_transform(X_train)
test_bigram_vectors = bigram_vectorizer.transform(X_test)

train_bigram_vectors.shape
test_bigram_vectors.shape

ch21 = SelectKBest(chi2, k=800)
# Transform your training and testing datasets accordingly
train_bigram_Kbest = ch21.fit_transform(train_bigram_vectors, Y_train)
test_bigram_Kbest = ch21.transform(test_bigram_vectors)

clf_ME = LogisticRegression(random_state=0, solver='lbfgs').fit(train_bigram_Kbest, Y_train)
predME = clf_ME.predict(test_bigram_Kbest)
pred = list(predME)
print(metrics.confusion_matrix(Y_test, pred))
print(metrics.classification_report(Y_test, pred))

model_svm = SVC(C=5000.0, gamma="auto", kernel='rbf')
clr_svm = model_svm.fit(train_bigram_Kbest, Y_train)   
predicted = clr_svm.predict(test_bigram_Kbest)
print(metrics.confusion_matrix(Y_test, predicted))
print(np.mean(predicted == Y_test) )
print(metrics.classification_report(Y_test, predicted))

###86 with 800/3000/8000/12000
###############################################################################################
###Bigram + Negation
bigram_vectorizer = TfidfVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
train_neg_bigram_vectors = bigram_vectorizer.fit_transform(X_nega_train)
test_neg_bigram_vectors = bigram_vectorizer.transform(X_nega_test)

train_neg_bigram_vectors.shape
test_neg_bigram_vectors.shape

ch21 = SelectKBest(chi2, k=8000)
# Transform your training and testing datasets accordingly
train_neg_bigram_Kbest = ch21.fit_transform(train_neg_bigram_vectors, Y_nega_train)
test_neg_bigram_Kbest = ch21.transform(test_neg_bigram_vectors)

clf_ME = LogisticRegression(random_state=0, solver='lbfgs').fit(train_neg_bigram_Kbest, Y_nega_train)
predME = clf_ME.predict(test_neg_bigram_Kbest)
pred = list(predME)
print(metrics.confusion_matrix(Y_nega_test, pred))
print(metrics.classification_report(Y_nega_test, pred))

model_svm = SVC(C=5000.0, gamma="auto", kernel='rbf')
clr_svm = model_svm.fit(train_neg_bigram_Kbest, Y_nega_train)   
predicted = clr_svm.predict(test_neg_bigram_Kbest)
print(metrics.confusion_matrix(Y_nega_test, predicted))
print(np.mean(predicted == Y_nega_test) )
print(metrics.classification_report(Y_nega_test, predicted))
###0.8672 with K = 2000/3000/5000/10000
###86 with 8000
################################################################################################
#With wordembddings

import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from os import getcwd, chdir
import pandas as pd

GLOVE_6B_100D_PATH = fpath+"\\Data\\glove.6B.100d.txt"

X_train = [t[0].split() for t in trainset]
X_test = [t[0].split() for t in testset]

X = X_train + X_test

Y_train = [t[1] for t in trainset]
Y_test = [t[1] for t in testset]

y = Y_train + Y_test

X[:2]
len(X)

X, y = np.array(X), np.array(y)
####################################################
#MeanEmbeddingVectorizer define the way to represent docs using word vectors

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec)>0:
            self.dim=len(word2vec[next(iter(glove_small))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
###################################################
#Prepare word embeddings from GLOVE
            
encoding="utf-8"
import numpy as np
with open(GLOVE_6B_100D_PATH, "rb") as lines:
    wvec = {line.split()[0].decode(encoding): np.array(line.split()[1:],dtype=np.float32)
               for line in lines}


# reading glove files, this may take a while
# we're reading line by line and only saving vectors
# that correspond to words from our data set
import struct 

glove_small = {}
all_words = set(w for words in X for w in words)

print(len(all_words))

with open(GLOVE_6B_100D_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0].decode(encoding)
        if (word in all_words):
            nums=np.array(parts[1:], dtype=np.float32)
            glove_small[word] = nums

###################################################
#Prepare word embeddings by training from dataset
model = Word2Vec(X, size=100, window=5, min_count=2, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}

w2v_Embedding = MeanEmbeddingVectorizer(w2v)
glove_Embedding = MeanEmbeddingVectorizer(glove_small)
#####Extend training dimensions 
X_w2v_train = w2v_Embedding.transform(X_train)
X_w2v_train.shape

X_glove_train = glove_Embedding.transform(X_train)
X_glove_train.shape

k_best = train_neg_bigram_Kbest.toarray()
'''
for sent,var in X_train:
    if sent has ? :
        question [var] = freq("?")
    else
        question [var] = 0
question=[][]
final_train = np.c_[X_w2v_train,X_glove_train,k_best,question]
'''
final_train = np.c_[X_w2v_train,X_glove_train,k_best]
final_train.shape

#####Extend testing dimensions 
X_w2v_test = w2v_Embedding.transform(X_test)
X_w2v_test.shape

X_glove_test = glove_Embedding.transform(X_test)
X_glove_test.shape

final_test = test_neg_bigram_Kbest.toarray();

final_test = np.c_[X_w2v_test,X_glove_test,final_test]
final_test.shape
#####Evaluate with models
clf_ME = LogisticRegression(random_state=0, solver='lbfgs').fit(final_train, Y_nega_train)
predME = clf_ME.predict(final_test)
pred = list(predME)
print(metrics.confusion_matrix(Y_nega_test, pred))
print(metrics.classification_report(Y_nega_test, pred))

model_svm = SVC(C=5000.0, gamma="auto", kernel='rbf')
clr_svm = model_svm.fit(final_train, Y_nega_train)   
predicted = clr_svm.predict(final_test)
print(metrics.confusion_matrix(Y_nega_test, predicted))
print(np.mean(predicted == Y_nega_test) )
print(metrics.classification_report(Y_nega_test, predicted))

######0.8688 with full vecs




