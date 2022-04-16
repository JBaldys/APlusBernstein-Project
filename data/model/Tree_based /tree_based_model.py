#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 19:20:50 2022

@author: anubhanagar
"""

import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import train_test_split from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import metrics






#col_names=['Score', 'output'],sep =",", header=None, sep='delimiter'
import numpy as np
pima=pd.read_csv ('/Users/anubhanagar/Desktop/APlusBernstein-Project/models/Tree_based /df_imputed_with_op.csv')
pima = pima.iloc[:,1:]
#pima = pima.select_dtypes(np.number)
pima.head()
np.size(pima)


X_train = pd.read_csv('/Users/anubhanagar/Desktop/APlusBernstein-Project/data/model/train_classification.csv')
X_train = X_train.iloc[:,1:]
y_train = X_train.loc[:, X_train.columns == "mom_1d_fwd_rel_d"]

X_test = pd.read_csv('/Users/anubhanagar/Desktop/APlusBernstein-Project/data/model/test_classification.csv')
X_test = X_test.iloc[:,1:]
y_test = X_test.loc[:, X_test.columns == "mom_1d_fwd_rel_d"]


X_train = X_train.iloc[: , :-3]
X_train.head()
X_test = X_test.iloc[: , :-3]
X_test
#y_train.head()
#split dataset in features and target variable



#feature_cols=['Score', 'output']
#X = pima.loc[:, pima.columns != "mom_1d_fwd_rel_d"]
#y = pima["mom_1d_fwd_rel_d"]# Target variable 
#y = pima.iloc[: , -1]
# Split dataset into training set and test set
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 


#0.5137857900318133 without feature trans


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
# Train Decision Tree Classifer
clf = clf.fit(X_train.values,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test.values) 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) 

# Visualising the Random Forest Regression results

# arrange for creating a range of values
# from min value of x to max
# value of x with a difference of 0.01
# between two consecutive values
import matplotlib.pyplot as plt
X_grid = np.arange(min(X_train), max(X_train), 0.01)

# reshape for reshaping the data into a len(X_grid)*1 array,
# i.e. to make a column out of the X_grid value				
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot for original data
plt.scatter(X_train, y_train, color = 'red')

# plot predicted data
plt.plot(X_grid, clf.predict(X_grid),color = 'black')
plt.title('Random Forest Regression')
plt.xlabel('DS Score')
plt.ylabel('Likeliness of accepting the offer')
plt.show()




import matplotlib.pyplot as plt
X_grid = np.arange(min(X_test), max(X_test), 0.01)

# reshape for reshaping the data into a len(X_grid)*1 array,
# i.e. to make a column out of the X_grid value				
X_grid = X_grid.reshape((len(X_grid), 1))

# Scatter plot for original data
plt.scatter(X_test, y_test, color = 'red')

# plot predicted data
plt.plot(X_grid, clf.predict(X_grid),color = 'black')
plt.title('Random Forest Regression')
plt.xlabel('DS Score')
plt.ylabel('Likeliness of accepting the offer')
plt.show()




