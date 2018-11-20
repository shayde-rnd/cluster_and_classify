"""
TF-IDF vector is a numeric representation of text - a vector in the size of the curpus where for each word
we have a weight: 1/(#number of occurrence of the word in the text) - each index in the vector represent a specific word
for instance, the following text:
  "Hi there, yes yes yes there"
  under the assumption that the curpus is:
  [... hi, ...  there, ... yes, ... home, ... what, ... ever, ...]
  [... 1, ...   0.5, ...   0.333, ... 0, ...  0, ...    0, ...]
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

import files_loader as filesLoader

NUM_OF_CLUSTERS = 5
# initialize a TF-IDF vector
print("*****************************")
print("**************Start***************")
tfidfvectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english")
curpus = filesLoader.loadFiles(dirPath="./articles")
print("**************After loading {0} files***************".format(len(curpus)))
"""
here we produce the vectors for each article
so we get a Matrix of "n X m" where:
n = number of articles
m = number of distinct words over the curpus
"""
import re
curpus = [re.sub(b"\n|\r", " ", text) for text in curpus]

X = tfidfvectorizer.fit_transform(curpus)
print("**************After TF-IDF calculation above all articles, received matrix of {0}***************".format(X.shape))
"""
initiate KMeans instance
init - the method of choosing the initial centroids
max_iter - maximum number of iterations, in case that didn't reach convergence before
""" 
km = KMeans(n_clusters=NUM_OF_CLUSTERS, init="k-means++", max_iter=100, n_init=1,verbose=False)
# Run KMeans algorithm
km.fit(X)
# initiate a numpy array to view statistics - we set 'return_counts' to view how many articles contained in each cluster
tup = np.unique(km.labels_, return_counts=True)
print("**************After KMeans clustering, statistics: {0}***************".format(tup))

# aggregate text per cluster articles
articleAggDic = {}

for i,clusterIdx in enumerate(km.labels_):
  article = curpus[i]
  if(clusterIdx not in articleAggDic.keys()):
    articleAggDic[clusterIdx] = article
  else:
    articleAggDic[clusterIdx] += article
print("After grouping all articles per cluster it belongs")
# import sentence and words tokenization
from nltk.tokenize import sent_tokenize, word_tokenize
# import stemmer
from nltk.stem.lancaster import LancasterStemmer
# stop words (the, an, where, which, ...)
from nltk.corpus import stopwords
# measure frequency
from nltk.probability import FreqDist
# defaultdict is a sub-class of dictionary - in case of get not-existing item it creates an empty entry instead of throw an exception
from collections import defaultdict
# punctuation list (. , : ! ...)
from string import punctuation
# n largest items in a list
from heapq import nlargest
# stopwords list
_stopwords = set(stopwords.words("english") + list(punctuation))

keywords = {}
counts={}
st=LancasterStemmer()
for clusterIdx in range(NUM_OF_CLUSTERS):
  # words of specific cluster
  word_sent = word_tokenize(articleAggDic[clusterIdx].lower())
  # filter stopwords
  word_sent=[word for word in word_sent if word not in _stopwords]
  # convert to lemma of each word
  word_sent=[st.stem(word) for word in word_sent]
  # list of tuples, each tuple contains a word and its frequency
  freq = FreqDist(word_sent)
  keywords[clusterIdx] = nlargest(100, freq, key=freq.get)
  counts[clusterIdx]=freq

print("After calculating frequencies and build top 100 frequent words")
unique_keys={}
for clusterIdx in range(NUM_OF_CLUSTERS):   
    other_clusters=list(set(range(NUM_OF_CLUSTERS))-set([clusterIdx]))
    keys_other_clusters = set([])
    for idx in other_clusters:
      keys_other_clusters = set(keywords[idx]).union(keys_other_clusters)
    unique=set(keywords[clusterIdx])-keys_other_clusters
    unique_keys[clusterIdx]=nlargest(20, unique, key=counts[clusterIdx].get)

print("After calculating unique most frequent words per cluster")
"""
import KNeighbors classifier
it accepts
"""
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=NUM_OF_CLUSTERS)
classifier.fit(X[:12000],km.labels_[:12000])

print("after training a classifier")
def classifyText(text):
  textAsTF_IDF_vec = tfidfvectorizer.transform([text])
  return classifier.predict(textAsTF_IDF_vec)


while True:
    user_input = input("Enter index between 12000 - 12056:")
    if user_input == "exit":
        break
    res = "?"
    castToInt= ''
    try:
      castToInt = int(user_input)
    except Exception as exp:
      pass
    
    if(isinstance(castToInt,int) and castToInt >= 12000 and castToInt < 12057):
      res = classifyText(curpus[castToInt])
    elif(isinstance(user_input,str)):
      res = classifyText(user_input)
    print("Res: {0}".format(res))