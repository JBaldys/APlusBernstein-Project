#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 19:25:17 2022

@author: anubhanagar
"""

#mlp 
# Kick off by importing libraries, and outlining the Iris dataset
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

X_train = pd.read_csv('/Users/anubhanagar/Desktop/APlusBernstein-Project/data/model/train_classification.csv')
X_train = X_train.iloc[:,1:]
y_train = X_train.loc[:, X_train.columns == "mom_1d_fwd_rel_d"]

X_test = pd.read_csv('/Users/anubhanagar/Desktop/APlusBernstein-Project/data/model/test_classification.csv')
X_test = X_test.iloc[:,1:]
y_test_mom = X_test.loc[:, X_test.columns == "mom_1d_fwd_rel_d"]
y_test_val = X_test.loc[:, X_test.columns == "mom_1d_fwd_rel_d"]




X_train = X_train.iloc[: , :-3]
X_train.head()
X_test = X_test.iloc[: , :-3]
X_test

# Feature scaling
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  
mlp.fit(X_train, y_train.values.ravel())  

predictions = mlp.predict(X_test) 
print(predictions)

print(confusion_matrix(y_test,predictions))  
print(classification_report(y_test,predictions))  