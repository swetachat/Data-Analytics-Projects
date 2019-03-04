
# coding: utf-8

# In[1]:


#!pip install BeautifulSoup4
import string
from bs4 import BeautifulSoup
import urllib.request
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import random
import collections
import pandas as pd

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import bigrams, trigrams               # NLTK has modules that can generate a list of bigrams and trigrams

# For model build
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.classify import MaxentClassifier       # For Logistics Regression model
# SVM model
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)


# In[2]:


# Define the bag_of_words function that will take a list and return
# a dictionary. The dictionary will contain each of the words in the list
# and it will have the value True assigned to each word
def bag_of_words(ngram_list):
    return dict([(ngram, True) for ngram in ngram_list])

# Functions to return a list of bigrams
def bigramReturner (list_of_words):
    bigramFeatureVector = []
    for item in bigrams(list_of_words):
        bigramFeatureVector.append(' '.join(item))
    return bag_of_words(bigramFeatureVector)
  
# Functions to return a list of trigrams
def trigramReturner (list_of_words):
    trigramFeatureVector = []
    for item in trigrams(list_of_words):
        trigramFeatureVector.append(' '.join(item))
    return bag_of_words(trigramFeatureVector)


# In[3]:


#a. Remove punctuation.
#b. Replace contractions.
#c. Convert all text to lower case.
#d. Remove stop words.

wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Removing negative words from english_stops, as this is  important keywords for negative sentiments
english_stops = set(stopwords.words('english'))
negate_words = ["doesn't","no","not","didn't","don't","haven't","shouldn't","weren't","won't"]
english_stops = [word for word in english_stops if word not in negate_words]

def decontraction(review):    
    review = re.sub(r"won't", "will not", review)
    review = re.sub(r"can\'t", "can not", review)
    review = re.sub(r"ain't", "am not", review)
    review = re.sub(r"let's", "let us", review)    

    review = re.sub(r"n\'t", " not", review)
    review = re.sub(r"\'re", " are", review)
    review = re.sub(r"\'s", " is", review)
    review = re.sub(r"\'d", " would", review)
    review = re.sub(r"\'ll", " will", review)
    review = re.sub(r"\'t", " not", review)
    review = re.sub(r"\'ve", " have", review)
    review = re.sub(r"\'m", " am", review)
    review = re.sub(r"\'cause", "because", review)
    review = re.sub(r"\..", "", review)                   #replacing .. or ...
    
    return review


def cleanData(review):
    review = decontraction(review)
    words = word_tokenize(review)
    words = [word.lower() for word in words]
    words = [word for word in words if word not in english_stops]
    words = [word for word in words if word not in set(string.punctuation[1:])]

    stem_list = [stemmer.stem(word) for word in words]
    return stem_list


# In[4]:


# Scrape the reviews and the associated ratings
# Recode ratings as either positive (4-5) or negative (1-3

def scrapeReviewRating(hotelName):
    reviewList = []    
    prefix = "https://www.yelp.com/biz/"
    postfix = "?start="
    all_dataset = []
    
    # Headers will make it look like you are using a web browser
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.34'}
      
    for i in range(0,500,20):        
        url = prefix + hotelName + postfix +str(i)
        response = requests.get(url, headers=headers, verify=False).text
        
        # Create a soup object and find all 500 reviews
        soup = BeautifulSoup(response, "lxml")
        
        # Find the review blocks
        review_blocks = soup.find_all('div', 'review-content')
        
        for r in review_blocks:
            #scrape reviews and ratings
            review = r.p.text                        
            int_rating = float(r.find('div',{'class':'i-stars'}).get('title').split(' ')[0])
                     
            # Recode 1-3 rating as neg and 4-5 as pos
            rating = "neg" if int_rating < 4 else "pos"
            
            all_dataset.append((rating,review))       #create list of tuples with (string,string)
    print("Dataset Created")      
    
    return all_dataset


# In[5]:


# Build  naïve Bayes, decision tree, logistic regression and SVM models

def getMatrix(classifier, test_set, datatype, classifierName):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
   
    #create ref and test datasets
    for i, (feats, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)
    
    df ={
            "Classifier":classifierName,
            "Data_type":datatype, 
            "Accuracy": nltk.classify.util.accuracy(classifier, test_set),
            "pos precision": precision(refsets['pos'], testsets['pos']),
            "pos recall": recall(refsets['pos'], testsets['pos']),
            "pos F-measure": f_measure(refsets['pos'], testsets['pos']),
            "neg precision": precision(refsets['neg'], testsets['neg']),
            "neg recall": recall(refsets['neg'], testsets['neg']),
            "neg F-measure": f_measure(refsets['neg'], testsets['neg'])
    }
    
    return df


# In[6]:


# Build model and Performance metrics
def buildModel(train_set, test_set, datatype):
    
    # Train the Naive Bayes model
    nb_classifier = NaiveBayesClassifier.train(train_set)
    metrics.append(getMatrix(nb_classifier,test_set,datatype,"Naive Bayes"))
   
    # Train a decision tree model
    dt_classifier = DecisionTreeClassifier.train(train_set, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=30)
    metrics.append(getMatrix(dt_classifier,test_set,datatype,"Decision Tree"))
   
    # Train Logistic regression model
    logit_classifier = MaxentClassifier.train(train_set, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)
    metrics.append(getMatrix(logit_classifier,test_set,datatype,"Logistics Regression"))
    
    # Train the SVM model
    SVM_classifier = SklearnClassifier(SVC(), sparse=False).train(train_set)
    metrics.append(getMatrix(SVM_classifier,test_set,datatype,"Support Vector Machine"))
    
    return metrics


# In[7]:


dataset = []
unigram_dataset = []
bigram_dataset = []
triigram_dataset = []
ngram_dataset = []

dataset = scrapeReviewRating("white-manna-hackensack")

# Create Unigram dataset
unigram_dataset = [(bag_of_words(cleanData(dataset[i][1])), dataset[i][0]) for i in range(len(dataset))] 

# Create bigram dataset
bigram_dataset = [(bigramReturner(cleanData(dataset[i][1])), dataset[i][0]) for i in range(len(dataset))] 

# Create trigram dataset
trigram_dataset = [(trigramReturner(cleanData(dataset[i][1])), dataset[i][0]) for i in range(len(dataset))] 

# Create ngram dataset
ngram_dataset =unigram_dataset + bigram_dataset + trigram_dataset


# In[1]:


metrics = []

# Shuffle Dataset
random.shuffle(unigram_dataset)
random.shuffle(bigram_dataset)
random.shuffle(trigram_dataset)
random.shuffle(ngram_dataset)

train_unigramDataset =[]

# Split dataset into 70/30 and create the training and test dataset
train_unigramDataset, test_unigramDataset = unigram_dataset[:int(len(unigram_dataset)*.70)], unigram_dataset[int(len(unigram_dataset)*.70):]
train_bigramDataset, test_bigramDataset = bigram_dataset[:int(len(bigram_dataset)*.70)], bigram_dataset[int(len(bigram_dataset)*.70):]
train_trigramDataset, test_trigramDataset = trigram_dataset[:int(len(trigram_dataset)*.70)], trigram_dataset[int(len(trigram_dataset)*.70):]
train_ngramDataset, test_ngramDataset = ngram_dataset[:int(len(ngram_dataset)*.70)], ngram_dataset[int(len(ngram_dataset)*.70):]

# Build Model/Metrics using unigrams
buildModel(train_unigramDataset, test_unigramDataset,"Unigrams")

# Build Model/Metrics using bigrams
buildModel(train_bigramDataset, test_bigramDataset,"Bigrams")

# Build Model/Metrics using trigrams
buildModel(train_trigramDataset, test_trigramDataset,"Trigrams")

# Build Model/Metrics using nigrams
buildModel(train_ngramDataset, test_ngramDataset,"Ngrams")

metrics_df =pd.DataFrame(metrics)
metrics_df


# Summarizing Model Performance Metrics.
# 
# Naive Bayes model produces the best negative review recall for bigrams, trigrams and ngrams.
# 
# Also, Naive Bayes gives 100% neg review recall for unigrams, which shouldn't be possible for a model. One of the reason could be imbalance dataset. Also, using unigram, Naive Bayes should have produce high TP and very low FN, which makes model 100% recall.
# 
# 
