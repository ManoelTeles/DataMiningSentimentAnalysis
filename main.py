from extract import extract
from sentimentAnalysis import sentimentAnalysis



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



if __name__ == '__main__':
    '''
        Funções que tratam a extração dos Tweets
    '''

    ext = extract.Extract()
    ext.QuerySearch = 'dafiti sapatos'
    ext.DateIni = "2019-11-01"
    ext.DateEnd = "2019-11-30"
    ext.QtyTweets = 20

    # # tweets = ext.get_OldTweets()
    # # df_tweets = ext.mount_df_tweets(tweets)
    # # ext.write_tweets(df_tweets)    
    # df_tweets_train = ext.get_csv('tweets_train2', ';')
    # df_tweets = ext.get_csv('tweets', '\t')

    # '''
    #     Analise de Sentimento - Limpeza
    # '''
    # sent = sentimentAnalysis.SentimentAnalysis()

    # ## test - new tweets
    # df_tweets = sent.removal_punctuations(df_tweets, 'text')
    # df_tweets = sent.str_to_lower(df_tweets, 'text')
    # df_tweets = sent.removal_stopwords(df_tweets, 'text')
    # df_tweets = sent.set_normalization(df_tweets, 'text')
    # df_tweets = df_tweets[['username', 'text']]    

    # ## train - new tweets
    # df_tweets_train = sent.removal_punctuations(df_tweets_train, 'text')
    # df_tweets_train = sent.str_to_lower(df_tweets_train, 'text')
    # df_tweets_train = sent.removal_stopwords(df_tweets_train, 'text')
    # df_tweets_train = sent.set_normalization(df_tweets_train, 'text')
    # df_tweets_train = df_tweets_train[['username', 'text', 'sentiment']] 












    def text_processing(tweet):
        
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

    train_tweets = ext.get_csv('tweets_train2', ';')
    test_tweets = ext.get_csv('tweets', '\t')
    
    train_tweets = train_tweets[['sentiment','text']]
    test = test_tweets['text']

    X = train_tweets['text']
    y = train_tweets['sentiment']
    test = test_tweets['text']

    from sklearn.model_selection import train_test_split
    msg_train, msg_test, label_train, label_test = train_test_split(train_tweets['text'], train_tweets['sentiment'], test_size=0.2)

    pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
    pipeline.fit(msg_train,label_train)

    predictions = pipeline.predict(msg_test)

    print(predictions)

    print(classification_report(predictions,label_test))
    print ('\n')
    print(confusion_matrix(predictions,label_test))
    print(accuracy_score(predictions,label_test))