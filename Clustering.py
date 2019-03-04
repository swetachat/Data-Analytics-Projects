
# coding: utf-8

# In[11]:


import time
from bs4 import BeautifulSoup
import urllib.request
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
#!pip install BeautifulSoup4


# In[12]:


# Create ReviewList for 20 different hotel using Yelp

def buildSynopses(hotelNameList):
    reviewList = []    
    prefix = "https://www.yelp.com/biz/"
        
    # Headers will make it look like you are using a web browser
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.34'}
        
    for hotelName in hotelNameList:
        reviews =[]
        url = prefix + hotelName  
        response = requests.get(url, headers=headers, verify=False).text
        
        # Create a soup object and find all 20 reviews
        soup = BeautifulSoup(response, "lxml")
        reviews = soup.find_all('p', attrs={'lang':'en'})
              
        # Concatenate all the 20 reviews into a string and put it in a list
        reviewList.append(' '.join(str(i.text) for i in reviews))   
               
    print("Synopses Created")        
    
    return reviewList


# In[13]:


#creating hotelnamelist to retrieve hotel review

titles = []
hotelNameList = []

# Building list 20 different restaurants with name and loc offering at least 4 different cuisines
#--Italian
hotelNameList.append("solaris-restaurant-hackensack")
hotelNameList.append("la-couronne-restaurant-montclair")
hotelNameList.append("tutta-pesca-hoboken")
hotelNameList.append("maggianos-little-italy-hackensack")
hotelNameList.append("bensi-of-hasbrouck-heights-hasbrouck-heights")

#--Greek
hotelNameList.append("stamna-greek-taverna-little-falls-4")
hotelNameList.append("beyond-pita-montclair")
hotelNameList.append("jackies-grillette-montclair")
hotelNameList.append("main-street-taverna-belleville")
hotelNameList.append("greek-taverna-montclair-4")

#--American
hotelNameList.append("laboratorio-kitchen-montclair-5")
hotelNameList.append("de-novo-european-pub-montclair")
hotelNameList.append("broughton-grill-montclair")
hotelNameList.append("uptown-596-montclair")
hotelNameList.append("pig-and-prince-montclair-4")

#--Chinese
hotelNameList.append("wahchung-chinese-restaurant-montclair")
hotelNameList.append("tasty-fusion-lyndhurst-2")
hotelNameList.append("pandan-asian-cuisine-and-delicacies-bloomfield")
hotelNameList.append("veggie-heaven-montclair")
hotelNameList.append("lucky-star-bloomfield")


# Create titles as 'cuisine:Restaurants name' for dendrogram
titles.append("Italian:Solaris-Restaurant")
titles.append("Italian:La-Couronne-Restaurant")
titles.append("Italian:Tutta-Resca")
titles.append("Italian:Maggianos-Little-Italy")
titles.append("Italian:Bensi-of-Hasbrouck-Heights")

titles.append("Greek:Btamna-Greek-Taverna")
titles.append("Greek:Beyond-Pita")
titles.append("Greek:Jackies-Grillette")
titles.append("Greek:Main-Street-Taverna")
titles.append("Greek:Greek-Taverna")

titles.append("American:Laboratorio-Kitchen")
titles.append("American:De-Novo-European-Pub")
titles.append("American:Broughton-Grill")
titles.append("American:Uptown-596")
titles.append("American:Pig-and-Prince")

titles.append("Chinese:Wahchung-Chinese-Restaurant")
titles.append("Chinese:Tasty-Fusion")
titles.append("Chinese:Pandan-Asian-Cuisine-&-Delicacies")
titles.append("Chinese:Veggie-Heaven")
titles.append("Chinese:Lucky-Star")


# In[16]:


def tokenize_and_stem(text):
    
    stemmer = SnowballStemmer("english")
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


# In[17]:


#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

synopses = buildSynopses(hotelNameList)  # list of 20 string created

get_ipython().magic('time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses')

print(tfidf_matrix.shape)


# In[18]:


from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)


# In[19]:


import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

plt.tick_params(    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('Restaurant_clusters.png', dpi=300) #save figure as Restaurant_clusters


# <img src='Restaurant_clusters.png'>

# <b>Do restaurants offering the same cuisine cluster together?</B>
# 
# Most of the restaurant offering same cuisine cluster together. But we can see one of the restaurant offering Greek cuisine, cluster with the restaurants with American cuisines and one of the American get cluster with Italian.
# 
# Also, cluster of Greek cuisines shows similarities with American_Italian cluster.
# 
# 
