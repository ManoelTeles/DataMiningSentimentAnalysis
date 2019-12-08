#Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk import PunktSentenceTokenizer

class SentimentAnalysis:

    def text_processing(self, tweet):
        
        #Generating the list of words in the tweet (hastags and other punctuations removed)
        def form_sentence(tweet):
            tweet_blob = TextBlob(tweet)
            return ' '.join(tweet_blob.words)
        new_tweet = form_sentence(tweet)
        
        #Removing stopwords and words with unusual symbols
        def no_user_alpha(tweet):
            tweet_list = [ele for ele in tweet.split() if ele != 'user']
            clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
            clean_s = ' '.join(clean_tokens)
            clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
            return clean_mess
        no_punc_tweet = no_user_alpha(new_tweet)
        
        #Normalizing the words in tweets 
        def normalization(tweet_list):
            stemmer = SnowballStemmer("portuguese")
            normalized_tweet = []
            for word in tweet_list:
                normalized_text = stemmer.stem(word)
                normalized_tweet.append(normalized_text)
            return normalized_tweet    
                
        return normalization(no_punc_tweet)

    def get_df_predicitions(self, test_tweets, predictions):    
        test_tweets['predictions'] = None
        for num, i in enumerate(predictions):
            test_tweets['predictions'][num] = i
        return test_tweets                