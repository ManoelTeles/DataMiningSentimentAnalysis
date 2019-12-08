import os

import GetOldTweets3 as got
import pandas as pd


class Extract:

    def __init__(self):
        self.files_dir = os.path.dirname(os.path.abspath(__file__))        
        self.QuerySearch = ''
        self.DateIni = ''
        self.DateEnd = ''
        self.QtyTweets = 5

    @property
    def QuerySearch(self):
         return self._QuerySearch

    @QuerySearch.setter
    def QuerySearch(self, value):
         self._QuerySearch = value

    @property
    def DateIni(self):
         return self._DateIni

    @DateIni.setter
    def DateIni(self, value):
         self._DateIni = value

    @property
    def DateEnd(self):
         return self._DateEnd

    @DateEnd.setter
    def DateEnd(self, value):
         self._DateEnd = value

    @property
    def QtyTweets(self):
         return self._QtyTweets

    @QtyTweets.setter
    def QtyTweets(self, value):
         self._QtyTweets = value

    def get_OldTweets(self):
        tweetCriteria = got.manager.TweetCriteria().setQuerySearch(self.QuerySearch)\
                                                .setSince(self.DateIni)\
                                                .setUntil(self.DateEnd)\
                                                .setMaxTweets(self.QtyTweets)
        tweets = got.manager.TweetManager.getTweets(tweetCriteria)
        return tweets

    def mount_df_tweets(self, tweets):
        json_tweets = []
        for t in tweets:
            obj_tweets = {"id":"",
                "username":"",
                "text":"",
                "retweets":"",
                "geo":""}

            obj_tweets["id"] = t.id
            obj_tweets["username"] = t.username
            obj_tweets["text"] = t.text
            obj_tweets["retweets"] = t.retweets
            obj_tweets["geo"] = t.geo
            json_tweets.append(obj_tweets)        
        df = pd.DataFrame(json_tweets)
        return df
    
    def write_tweets(self, df):        
        csv_name = '{}/files/tweets.csv'.format(self.files_dir)
        df.to_csv(csv_name, sep='\t')

    def get_csv(self, name, sep):
        csv_name = '{}/files/{}.csv'.format(self.files_dir,name)
        df = pd.read_csv(csv_name, sep=sep)
        return df
