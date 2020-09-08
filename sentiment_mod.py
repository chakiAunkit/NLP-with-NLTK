#!/usr/bin/env python
# coding: utf-8

# File: Sentiment_mod.py

# In[42]:


import nltk
import random
#from nltk.corpus import movie_reviews
import warnings
warnings.filterwarnings('ignore')
from nltk.tokenize import word_tokenize
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression,SGDClassifier

from nltk.classify import ClassifierI
from statistics import mode


# In[43]:


nltk.data.path.append("C:/Anaconda/pkgs/nltk-3.4.5-py37_0/Lib/site-packages/nltk/")


# In[44]:


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


# In[45]:


documents_f = open("pickle_algos/documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


# In[46]:


word_features5k_f = open("pickle_algos/word_features5k.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


# In[47]:


def find_features(document):
    words = nltk.word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


# In[52]:


#find_features("Hey. It is so beautiful.")


# In[48]:


import os


# In[49]:


if os.path.getsize("pickle_algos/featuresets5k.pickle") > 0:
    with open("pickle_algos/featuresets5k.pickle", "rb") as f:
        unpickler = pickle.Unpickler(f)
        featureset = unpickler.load()


# In[50]:


random.shuffle(featureset)


# In[51]:


training_set = featureset[:9500]
testing_set = featureset[9500:]


# In[53]:


if os.path.getsize("pickle_algos/originalnaivebayes5k.pickle") > 0:
    with open("pickle_algos/originalnaivebayes5k.pickle", "rb") as f:
        unpickler = pickle.Unpickler(f)
        classifier = unpickler.load()


# In[54]:


if os.path.getsize("pickle_algos/MNB_classifier5k.pickle") > 0:
    with open("pickle_algos/MNB_classifier5k.pickle", "rb") as f:
        unpickler = pickle.Unpickler(f)
        MNB_classifier = unpickler.load()


# In[55]:


if os.path.getsize("pickle_algos/BernoulliNB_classifier5k.pickle") > 0:
    with open("pickle_algos/BernoulliNB_classifier5k.pickle", "rb") as f:
        unpickler = pickle.Unpickler(f)
        Bernoulli_classifier = unpickler.load()


# In[56]:


if os.path.getsize("pickle_algos/LinearSVC_classifier5k.pickle") > 0:
    with open("pickle_algos/LinearSVC_classifier5k.pickle", "rb") as f:
        unpickler = pickle.Unpickler(f)
        LinearSVC_classifier = unpickler.load()


# In[57]:


if os.path.getsize("pickle_algos/LogisticRegression_classifier5k.pickle") > 0:
    with open("pickle_algos/LogisticRegression_classifier5k.pickle", "rb") as f:
        unpickler = pickle.Unpickler(f)
        LogisticRegression_classifier = unpickler.load()


# In[58]:


voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  Bernoulli_classifier,
                                  LinearSVC_classifier,
                                  LogisticRegression_classifier)

def sentiment(text):
    feats = find_features(text)
    
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)


# In[ ]:




