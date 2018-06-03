# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:02:35 2018

@author: Aleksei
"""

from sklearn import tree, svm, discriminant_analysis
import numpy as np
clf = tree.DecisionTreeClassifier()
clf1 = svm.SVC()
clf2 = discriminant_analysis.QuadraticDiscriminantAnalysis()


# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


clf = clf.fit(X, Y)
clf1 = clf1.fit(X, Y)
clf2 = clf2.fit(X, Y)

prediction = clf.predict([[190, 70, 43], [88,50,40]])
prediction1 = clf1.predict([[190, 70, 43]])
prediction2 = clf2.predict([[190, 70, 43]])

# CHALLENGE compare their reusults and print the best one!

print(prediction, prediction1, prediction2)
index = np.argmax([0.95, 0.4, 0.2])
print (index)