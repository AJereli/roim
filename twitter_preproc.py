import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

DATASET_ENCODING = "ISO-8859-1"


def import_tweets(filename, header = None):
    tweet_dataset = pd.read_csv(filename, encoding=DATASET_ENCODING, header = header)
    tweet_dataset.columns = ['sentiment','id','date','flag','user','text']
    for i in ['flag','id','user','date']: del tweet_dataset[i] # or tweet_dataset = tweet_dataset.drop(["id","user","date","user"], axis = 1)
    tweet_dataset.sentiment = tweet_dataset.sentiment.replace(4,1)
    return tweet_dataset


def preprocess_tweet(tweet):
    tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER', tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    return tweet


def feature_extraction(data, method = "tfidf"):
    #arguments: data = all the tweets in the form of array, method = type of feature extracter
    #methods of feature extractions: "tfidf" and "doc2vec"
    if method == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfv=TfidfVectorizer(sublinear_tf=True, stop_words="english")
        features=tfv.fit_transform(data)
    else:
        return "Incorrect inputs"
    return features


def preprocess(text, stem=False):
    # nltk.download('stopwords')
    stop_words = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    text = str(text)
    text.lower()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('@[^\s]+', 'AT_USER', text)
    text = re.sub('[\s]+', ' ', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return text
