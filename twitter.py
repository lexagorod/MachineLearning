# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:59:43 2018

@author: Aleksei


wik = TextBlob("sample text")
print(wik.sentiment.polarity, wiki.words)
"""
import tweepy
import pandas as pd
from textblob import TextBlob

consumer_key = ''
consumer_secret = ''

access_token = ''
access_token_secret = ''


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
sentim = []
textp = []
textn = []
public_tweets = api.search('Trump')

for tweet in public_tweets:
    analysis = TextBlob(tweet.text)
    if analysis.sentiment.polarity > 0.1:
       textp.append(tweet.text)
    else:
        textn.append(tweet.text)
d = {'Positive':textp, 'Negative':textn}
df = pd.DataFrame.from_dict(d, orient='index')
dk=df.transpose()
print(dk[:2][1])
dk.to_csv('example.csv')