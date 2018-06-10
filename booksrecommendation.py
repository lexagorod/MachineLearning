# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:24:25 2018

@author: Aleksei
"""

import numpy as np
from lightfm import LightFM
from pandas import read_csv
from scipy.sparse import coo_matrix
from lightfm.evaluation import precision_at_k




#fetch data books
user_id = read_csv('BX-Users.csv', error_bad_lines=False, encoding='latin-1', sep=';')
books = read_csv('BX-Books.csv',  error_bad_lines=False, encoding='latin-1', sep=';')
rating = read_csv ('BX-Book-Ratings.csv', error_bad_lines=False, encoding='latin-1', sep=';')
authorbook = []
#create dictionary to match ISBN number from books and rating csvs
ratin_book_dict = {k: v for (v, k) in enumerate(set(rating['ISBN']), start=1)}
#substitute ISBN codes with numbers from dictionary
ratin_book_ISBNnum = [ratin_book_dict[el] for el in rating['ISBN']]
ratin_book_ISBNnumbook = []
for el in range(len(books['ISBN'])):
    if books['ISBN'][el] in ratin_book_dict:
        ratin_book_ISBNnumbook.append(ratin_book_dict[books['ISBN'][el]])
        tuple1 = (books['Book-Title'][el],books['Book-Author'][el])
        authorbook.append(tuple1)
#create dictionary with keys as numbers instead of ISBN code from ratin_book_dictionary
ISBNbook = dict(zip(ratin_book_ISBNnumbook, authorbook))

 
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
Rate, User, ISBNN = desirable_sample(9, rating['Book-Rating'], rating['User-ID'], ratin_book_ISBNnum)
print(len(ISBNN))
#choose training sample size
a = int(len(Rate)/7)
print(a)
Rate_train, User_train, ISBN_train = Rate[:a], User[:a], ISBNN[:a]
print(len(ISBN_train))
#set training and full sparse matrix
coo = coo_matrix((Rate_train,(User_train ,ISBN_train)))
coofull = coo_matrix(( Rate, (User , ISBNN)))

# We return 2 matrices and book info dictionary
Alldata = {
        'train' : coo,
        'all' : coofull,
        'books': ISBNbook
        }
print(repr(Alldata['train']))
print(repr(Alldata['all']))

#print(itemgetter(1)(Alldata['books']))


#choose best model for training
def determine_best_model(data):
     models = ['bpr', 'warp',  'warp-kos']
     train_auc_max = 0
     modelmax = LightFM()
     for i in models:
       model = LightFM(loss='{}'.format(i))
       model.fit(data['train'], epochs=30, num_threads=1)
       train_auc = precision_at_k(model, data['train'], k=10).mean()
       print('Precision: train %.2f.' % (train_auc))  
       if train_auc > train_auc_max:
          train_auc_max = train_auc
          modelmax = model  
     return modelmax
 

def sample_recommendation(model, data, user_ids):

    #number of users and books in training data
    n_users, n_items = data['train'].shape
    for user_id in user_ids:
    
        #books they already like
        known_positives = [data['books'].get(key) for key in data['train'].tocsr()[user_id].indices]
        print(known_positives[:3])
        #books our model predicts they will like
        scores = model.predict(user_id, data['train'].tocsr().indices)
        #rank them in order of most liked to least
        top_items = [data['books'].get(key) for key in data['train'].tocsr().indices[np.argsort(-scores)]]

        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for i in range(3):
            print(known_positives[i])

        print("     Recommended:")

        for x in range(3):
            print(top_items[x])
           

sample_recommendation(determine_best_model(Alldata), Alldata, [11676])
