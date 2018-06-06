# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:24:25 2018

@author: Aleksei
"""

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
from pandas import read_csv
from scipy.sparse import coo_matrix
from lightfm.evaluation import precision_at_k



#fetch data books
user_id = read_csv('BX-Users.csv', error_bad_lines=False, encoding='latin-1', sep=';')
books = read_csv('BX-Books.csv',  error_bad_lines=False, encoding='latin-1', sep=';')
rating = read_csv ('BX-Book-Ratings.csv', error_bad_lines=False, encoding='latin-1', sep=';')
authorbook = []
ratin_book_dict = {k: v for (v, k) in enumerate(set(rating['ISBN']), start=1)}
ratin_book_ISBNnum = [ratin_book_dict[el] for el in rating['ISBN']]
ratin_book_ISBNnumbook = []
for el in range(len(books['ISBN'])):
    if books['ISBN'][el] in ratin_book_dict:
        ratin_book_ISBNnumbook.append(ratin_book_dict[books['ISBN'][el]])
        tuple1 = (books['Book-Title'][el],books['Book-Author'][el])
        authorbook.append(tuple1)
ISBNbook = dict(zip(ratin_book_ISBNnumbook, authorbook))
print(ISBNbook[1], ratin_book_dict['0060182946'])
""" 
#set desirable min score
def desirable_sample(score, BookRate, User_id, ISBNN):
    Rate = []
    User = []
    ISBN = []
    for i in range(len(rating['Book-Rating'])-1):
        if  rating['Book-Rating'][i] >= score:
            Rate.append(BookRate[i])
            User.append(User_id[i])
            ISBN.append(ISBNN[i])
    return Rate, User, ISBN
#store valuse with above desirable score
Rate, User, ISBNN = desirable_sample(5, rating['Book-Rating'], rating['User-ID'], ratin_book_ISBNnum)

#choose training sample size
a = int(len(Rate)/10)
print(a)
Rate_train, User_train, ISBN_train = Rate[:a], User[:a], ISBNN[:a]

#set training and full sparse matrix
coo = coo_matrix((Rate_train,(User_train ,ISBN_train)))
coofull =coo_matrix(( Rate, (User , ISBNN)))

# We return the matrix, the artist dictionary and the amount of users
Alldata = {
        'train' : coo,
        'all' : coofull,
        'books': ISBNbook
        }
print(repr(Alldata['train']))
print(repr(Alldata['all']))

#choose best model for training
def determine_best_model(data):
     models = ['bpr', 'warp',  'warp-kos']
     train_auc_max = 0
     modelmax = LightFM()
     for i in models:
       model = LightFM(loss='{}'.format(i))
       model.fit(data['train'], epochs=15, num_threads=1)
       train_auc = precision_at_k(model, data['all'], k=10).mean()
       print('Precision: train %.2f.' % (train_auc))  
       if train_auc > train_auc_max:
           train_auc_max = train_auc
           modelmax = model  
     return modelmax
 

def sample_recommendation(data, user_ids):

    #number of users and movies in training data
    n_users, n_items = data['all'].shape

    for user_id in user_ids:
    
        #movies they already like
        known_positives = data['books'][data['all'].tocsr()[user_id].indices]
        print(known_positives)
    
        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        #rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]

        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)
           
#sample_recommendation(Alldata)
            
#determine_best_model(Alldata)
"""
#sample_recommendation(Alldata, [276828, 276832])