#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:45:33 2022

@author: anubhanagar
"""

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import train_test_split from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics


X_train = pd.read_csv('/Users/anubhanagar/Desktop/APlusBernstein-Project/data/After_feature_eng/X_train_transformed.csv')
y_train = pd.read_csv('/Users/anubhanagar/Desktop/APlusBernstein-Project/data/After_feature_eng/y_train.csv')

X_test = pd.read_csv('/Users/anubhanagar/Desktop/APlusBernstein-Project/data/After_feature_eng/X_test_transformed.csv')
y_test = pd.read_csv('/Users/anubhanagar/Desktop/APlusBernstein-Project/data/After_feature_eng/y_test.csv')


import numpy as np
for i in X_train:
    a = X_train[X_train[i]== np.inf]
    print(a)


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train.values,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test.values) 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) 