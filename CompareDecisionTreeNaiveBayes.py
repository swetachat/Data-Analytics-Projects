
# coding: utf-8

# <b> Comparision between Naive Bayes Model and Decision Tree Model for Hotel Reivew - Trip Advisor. 

# In[1]:


import time
import string
import random
import csv
import nltk
import collections

from selenium import webdriver
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk.metrics.scores import (accuracy, precision, recall, f_measure)
from nltk.classify import DecisionTreeClassifier
from nltk.stem import PorterStemmer


# In[2]:


wordnet_lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Removing negative words from english_stops, as this is  important keywords for negative sentiments
english_stops = set(stopwords.words('english'))
negate_words = ["doesn't","no","not","didn't","don't","haven't","shouldn't","weren't","won't"]
english_stops = [word for word in english_stops if word not in negate_words]

def bow_features(review):
    words = word_tokenize(review)
    words = [word.lower() for word in words]
    words = [word for word in words if word not in stopwords.words("english")]
    words = [word for word in words if word not in set(string.punctuation[1:])]

    stem_list = [stemmer.stem(word) for word in words]
    my_dict = dict([(word, True) for word in stem_list])
    return my_dict


# In[3]:


def buildDataSet(prefix, hotelName):
    # Use chrome webdriver
    browser = webdriver.Chrome()

    dataset = []
    
    for i in range(0,10,5):

        url = prefix + str(i) +"-"+ hotelName + ".html"
        browser.get(url)
        
        time.sleep(15)

        # Find and click the "more" links
        more_links = browser.find_elements_by_xpath("//span[@class='taLnk ulBlueLinks']")
        for l in more_links:
            try:
                l.click()
            except:
                pass

        # Use BeautifulSoup to parse the webpages
        html = browser.page_source
        soup = BeautifulSoup(html, "lxml")

        # Find the review blocks
        review_blocks = soup.find_all('div', 'reviewSelector')

        for r in review_blocks:
            int_rating = int(r.find('span','ui_bubble_rating')['class'][1].split('_')[1])//10

            # Convert 1-3 rating as neg and 4-5 as pos
            rating = "neg" if int_rating < 4 else "pos"

            review = r.p.text
            dataset.append((rating,review))      #create list of tuples with (string,string)
           
    browser.quit()       
    
    #creating list of above created tuples with each tuples as (dict,label)
    featured_dataset =[(bow_features(dataset[i][1]), dataset[i][0]) for i in range(len(dataset))]   
    print("Dataset Created") 
    
    return featured_dataset


# In[4]:


def buildModel(classifier, train_set,test_set):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
   
    #create ref and test datasets
    for i, (feats, label) in enumerate(test_set):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print('Accuracy:', nltk.classify.util.accuracy(classifier, test_set))
    print('pos precision:', precision(refsets['pos'], testsets['pos']))
    print('pos recall:', recall(refsets['pos'], testsets['pos']))
    print('pos F-measure:', f_measure(refsets['pos'], testsets['pos']))
    print('neg precision:', precision(refsets['neg'], testsets['neg']))
    print('neg recall:', recall(refsets['neg'], testsets['neg']))
    print('neg F-measure:', f_measure(refsets['neg'], testsets['neg']))
    


# In[5]:


# Building model for hotel 1
hotel1Prefix = "https://www.tripadvisor.com/Hotel_Review-g60763-d584986-Reviews-or"
hotel1 ="Hotel_Central_Fifth_Avenue_New_York-New_York_City_New_York"

#randamize the list of tuples
featured_dataset = buildDataSet(hotel1Prefix,hotel1)
random.shuffle(featured_dataset)
    
#create the training and test set
train_set, test_set = featured_dataset[:int(len(featured_dataset)*.75)], featured_dataset[int(len(featured_dataset)*.75):]

print(type(train_set))
print(len(train_set))
# Train the Naive Bayes model
nb_classifier = NaiveBayesClassifier.train(train_set)

# Train a decision tree model
dt_classifier = DecisionTreeClassifier.train(train_set, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=30)

print("--------------- Naive Bayes -----------------------")
buildModel(nb_classifier,train_set,test_set)

print("--------------- Decision Tree -----------------------")
buildModel(dt_classifier,train_set,test_set)


# <hr>
# <B> Better Model : Decision Tree </B>
# <p>Since this is Hotel review, useful for the regional manager to find the customer feedbacks. 
# The actual negative reviews, will help the management to find out the areas where the corrective actions needs to be taken, to improve the customer services. 
# 
# The false positive are the one which are negative reviews in actual but predicted as positive reviews. The less FP the more is the precision. Model with more precision will provide more benefits.</p>
# 
# <b>Precision = TP/(TP +FP)</b>
# 
# Lesser FP, more is the precision, better is the model.
# 
# neg Precision says the files that are neg but are incorrectly identified. 
# 
# For Naive Bayes the neg precision is 50% while in Decision tree the neg precision is 64%. This means there are less FP for neg class in Decision trees model.
# 
# So, considering Decision as our better model, we will use it to evaluate our second hotel.

# In[96]:


# Using Naive Bayes for second Hotel

hotel2Prefix = "https://www.tripadvisor.com/Hotel_Review-g60763-d99766-Reviews-or"
hotel2 = "The_Roosevelt_Hotel-New_York_City_New_York"

#randamize the list of tuples
featured_dataset = buildDataSet(hotel2Prefix,hotel2)
random.shuffle(featured_dataset)
    
#create the training and test set
train_set, test_set = featured_dataset[:int(len(featured_dataset)*.75)], featured_dataset[int(len(featured_dataset)*.75):]

# Train a decision tree model
dt_classifier = DecisionTreeClassifier.train(train_set, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=30)

print("--------------- Output -----------------------")
buildModel(dt_classifier,train_set,test_set)


# <b><u>Results Summary</u></b>
# 
# <B>2nd hotel</B> 
# <br>Accuracy: 0.7066666666666667
# <br>pos precision: 0.7368421052631579
# <br>pos recall: 0.7
# <br>pos F-measure: 0.717948717948718
# <br>neg precision: 0.6756756756756757
# <br>neg recall: 0.7142857142857143
# <br>neg F-measure: 0.6944444444444444
# 
# <B>1st hotel</B>
# <br>Accuracy: 0.6533333333333333
# <br>pos precision: 0.6557377049180327
# <br>pos recall: 0.8888888888888888
# <br>pos F-measure: 0.7547169811320754
# <br>neg precision: 0.6428571428571429
# <br>neg recall: 0.3
# <br>neg F-measure: 0.40909090909090906
# 
#  Accuracy is <b>increased</b>.
# <br>Pos Precision is <b>increased</b>.
# <br>Pos Recall is <b>decreased</b>.
# <br>Pos F-measure is <b>decreased</b>.
# <br>Neg Precision is <b>increased</b>.
# <br>Neg Recall is <b>increased</b>.
# <br>Neg F-measure is <b>increased</b>.
# 
# Higher precision means <b>less false positives</b>, while a lower precision means more <b>false positives</b>. 
# Higher recall means <b>less false negatives</b>, while lower recall means more <b>false negatives</b>. 
# 
# Increased neg precision denotes very <b>few false positive</b> for the neg class. But many files that are neg, are incorrectly classified. Increased neg recall, denotes <b> decrease false negative</b> for neg labels.
# 
# Decreased pos recall denotes <b>more false negative</b>for the pos class. There are files that are pos but are incorrectly classified. Increased pos precision, denotes <b>decreased false positive</b> for pos label.
# 
# 
# F-measure is combination of precision and recall.
# 

# In[97]:


nb_classifier.most_informative_features(n=10)


# In[98]:


nb_classifier.show_most_informative_features(n=8)

