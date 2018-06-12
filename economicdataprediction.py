# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 16:11:48 2018

@author: Aleksei
"""

import tweepy
import quandl
import numpy as np
import math
from textblob import TextBlob
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(7)



#insert your API keys
consumer_key = ''
consumer_secret = ''

access_token = ''
access_token_secret = ''

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#choose your company for sentiment analysis
public_tweets = api.search('Activision Blizzard')

def is_it_worthtodo_analysis():
    good_tweets = 0
    bad_tweets = 0
    for tweet in public_tweets:    
       analysis = TextBlob(tweet.text)
       if analysis.sentiment.polarity > 0:
          good_tweets=good_tweets+1
       else:
          bad_tweets=bad_tweets+1
    if good_tweets >= bad_tweets:
        worth = True
    print("good tweets: ", good_tweets, "bad_tweets: ",  bad_tweets)
    return worth

if not is_it_worthtodo_analysis():
    print ('This stock has bad sentiment, please choose another company, or rerun the script')
    sys.exit()
else: 
    print ('This stock has good sentiment, lets proceed')
   

#data collection
def get_data(company_data):
     data = quandl.get("{}".format(company_data), authtoken="", start_date="")
     prices = data['Previous Day Price'].values.astype('float32')
     dates = data.index.values.astype('M8[D]')
     return dates, prices
#choose your company from quandl
dates, prices = get_data("SSE/AIY")

#split into train and test sets
def split_data(data):
    train_size = int(len(data) * 0.8)
    train, test = data[0:train_size], data[train_size:len(data)]
    return train, test

#use window method or regression for perceptron
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)
#TO DO: use from sklearn.preprocessing import Imputer for filling missing data, 
#make function for capture future values and process them and on the basis of 
#all this add dates to predicted graph
#predict prices
def predict_prices(dates, prices):
    train_dates, test_dates = split_data(dates)
    train_prices, test_prices = split_data(prices)

    #window method, for regression look_back = 1
    look_back = 3
    
    trainX, trainY = create_dataset(train_prices, look_back)
    testX, testY = create_dataset(test_prices, look_back)

    model = Sequential()
    model.add(Dense(32,input_dim=look_back ,activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    #tested a lot of parameters, those are the best so far for this model
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    model.fit(trainX, trainY, epochs=150, batch_size = 40, verbose = 2)
    
    trainScore = model.evaluate(trainX, trainY,  verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(testX, testY,  verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))
    
    trainPredict = model.predict(trainX)
    testpredict = model.predict(testX)
    print(trainPredict.ravel())
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(prices)
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back-2:len(trainPredict)+look_back-2] = trainPredict.ravel()
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(prices)
    testPredictPlot[:] = np.nan
    testPredictPlot[len(trainPredict)+(look_back)+2:len(prices)-3] = testpredict.ravel()
    
    plt.axes([0,0, 3, 1])
    plt.plot(prices, label = 'actual prices')
    plt.plot(  trainPredictPlot, label = 'train')
    plt.plot( testPredictPlot, label = 'test')
    plt.legend()
    plt.show()

predict_prices(dates, prices)
    
