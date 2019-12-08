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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


class SentimentAnalysis:

    # def form_sentence(self, tweet):
    #     tweet_blob = TextBlob(tweet)
    #     return ' '.join(tweet_blob.words)

    # def no_user_alpha(self, tweet):
    #     tweet_list = [ele for ele in tweet.split() if ele != 'user']
    #     clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
    #     clean_s = ' '.join(clean_tokens)
    #     clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('portuguese')]
    #     return clean_mess        

    # def normalization(self, tweet_list):
    #     stemmer = SnowballStemmer("portuguese")
    #     normalized_tweet = []
    #     for word in tweet_list:
    #         normalized_text = stemmer.stem(word)
    #         normalized_tweet.append(normalized_text)
    #     return normalized_tweet    


    # def removal_punctuations(self, df, column):
    #     df[column] = df[column].apply(lambda x: self.form_sentence(x))
    #     return df

    # def str_to_lower(self, df, column):
    #     df[column] = df[column].apply(lambda x: x.lower())
    #     return df

    # def removal_stopwords(self, df, column):
    #     df[column] = df[column].apply(lambda x: self.no_user_alpha(x))
    #     return df

    # def set_normalization(self, df, column):
    #     df[column] = df[column].apply(lambda x: self.normalization(x))
    #     return df


    def text_processing(self, tweet):
        def form_sentence(self, tweet):
            tweet_blob = TextBlob(tweet)
            return ' '.join(tweet_blob.words)
        new_tweet = form_sentence(tweet)            

        def no_user_alpha(self, tweet):
            tweet_list = [ele for ele in tweet.split() if ele != 'user']
            clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
            clean_s = ' '.join(clean_tokens)
            clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('portuguese')]
            return clean_mess        

        def normalization(self, tweet_list):
            stemmer = SnowballStemmer("portuguese")
            normalized_tweet = []
            for word in tweet_list:
                normalized_text = stemmer.stem(word)
                normalized_tweet.append(normalized_text)
            return normalized_tweet
        return normalization(no_punc_tweet)            