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
# import sentence and words tokenization
from nltk.tokenize import sent_tokenize, word_tokenize
# stop words (the, an, where, which, ...)
from nltk.corpus import stopwords
# measure frequency
from nltk.probability import FreqDist
# sub-class of dictionary - in case of get not-existing item it creates an empty entry instead of throw an exception
from collections import defaultdict
# punctuation list (. , : ! ...)
from string import punctuation

import files_content_loader as fcl

NUM_OF_CLUSTERS = 10
# initialize a TF-IDF vector
print("*****************************")
print("**************Start***************")
tfidfvectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words="english")
curpus = fcl.loadFiles(dirPath="./articles")
print("**************After loading {0} files***************".format(len(curpus)))
"""
here we produce the vectors for each article
so we get a Matrix of "n X m" where:
n = number of articles
m = number of distinct words over the curpus
"""
X = tfidfvectorizer.fit_transform(curpus)
print("**************After TF-IDF calculation on all articles, matrix of {0}***************".format(X.shape))
"""
initiate KMeans instance
init - a way of choosing the initial centroids
max_iter - maximum number of iterations, in case that didn't reach convergence before
""" 
km = KMeans(n_clusters=NUM_OF_CLUSTERS, init="k-means++", max_iter=100, n_init=1,verbose=False)
# Run KMeans algorithm
km.fit(X)
# initiate a numpy array to view statistics - we set 'return_counts' to view how many articles contained in each cluster
tup = np.unique(km.labels_, return_counts=True)
print("**************After KMeans clustering, statistics: {0}***************".format(tup))

# aggregate text of per cluster articles
articleAggDic = {}

for i,clusterIdx in enumerate(km.labels_):
  article = curpus[i]
  if(clusterIdx not in articleAggDic.keys()):
    articleAggDic[clusterIdx] = article
  else:
    articleAggDic[clusterIdx] += article

