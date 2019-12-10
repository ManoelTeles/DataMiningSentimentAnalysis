from extract import extract
from sentimentAnalysis import sentimentAnalysis


#Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Preprocessing and Feature Engineering
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import train_test_split



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
    # # ext.write_tweets(df_tweets, 'tweets', '\t')    

    '''
        Analise de Sentimento - Limpeza
    '''
    sent = sentimentAnalysis.SentimentAnalysis()

    train_tweets = ext.get_csv('tweets_train2', ';')
    test_tweets = ext.get_csv('tweets', '\t')
    
    train_tweets = train_tweets[['sentiment','text']]
    test = test_tweets['text']

    X = train_tweets['text']
    y = train_tweets['sentiment']
    test = test_tweets['text']

    
    msg_train, msg_test, label_train, label_test = train_test_split(train_tweets['text'], train_tweets['sentiment'], test_size=0.2)

    pipeline = Pipeline([
        ('bow',CountVectorizer(analyzer=sent.text_processing)),  # strings to token integer counts
        ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
        ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
    ])
    pipeline.fit(msg_train,label_train)
    predictions = pipeline.predict(msg_test)
    df_predicitions = sent.get_df_predicitions(test_tweets, predictions)
    df_predicitions.drop(columns=['Unnamed: 0', 'geo'], inplace=True)
    ext.write_tweets(df_predicitions, 'predicitions', ',') 

    print(predictions)
    print(df_predicitions)
    print ('\n')
    print(classification_report(predictions,label_test))
    print ('\n')
    print(confusion_matrix(predictions,label_test))
    print(accuracy_score(predictions,label_test))