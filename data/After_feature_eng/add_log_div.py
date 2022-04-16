#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:15:02 2022

@author: anubhanagar
"""

import pandas as pd
import numpy as np 

X_train = pd.read_csv('/Users/anubhanagar/Desktop/APlusBernstein-Project/data/model/train_classification.csv')
X_train = X_train.iloc[:,1:]

y_train = X_train.iloc[:,-3:]

X_train = X_train.iloc[: , :-3]


X_test = pd.read_csv('/Users/anubhanagar/Desktop/APlusBernstein-Project/data/model/test_classification.csv')

X_test = X_test.iloc[:,1:]
y_test = X_test.iloc[:,-3:]

X_test = X_test.iloc[: , :-3]

# Calculate logarithm to base 10
for i in X_train:
    
    #print(data[col_name][j])
    X_train[i+ "/vix"] = pd.to_numeric(X_train[i])/pd.to_numeric(X_train["VIX"])
    X_train['log'+i] = np.log10(X_train[i])
    

X_train.to_csv("/Users/anubhanagar/Desktop/APlusBernstein-Project/data/After_feature_eng/X_train_transformed.csv")

#list(X_train.columns)

y_train.to_csv("/Users/anubhanagar/Desktop/APlusBernstein-Project/data/After_feature_eng/y_train.csv")

y_test.to_csv("/Users/anubhanagar/Desktop/APlusBernstein-Project/data/After_feature_eng/y_test.csv")

for i in X_test:
    
    #print(data[col_name][j])
    X_test[i+ "/vix"] = pd.to_numeric(X_test[i])/pd.to_numeric(X_test["VIX"])
    X_test['log'+i] = np.log10(X_test[i])
    
X_test.to_csv("/Users/anubhanagar/Desktop/APlusBernstein-Project/data/After_feature_eng/X_test_transformed.csv")
