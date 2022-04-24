#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 19:39:36 2022

@author: anubhanagar
"""


import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap
from matplotlib import pyplot as plt
#boston = load_boston()

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

#random forest feature importance 
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)

rf.feature_importances_

plt.barh(X_train.feature_names, rf.feature_importances_)


#2 permutaion based feature importance 
perm_importance = permutation_importance(rf, X_test, y_test)
sorted_idx = perm_importance.importances_mean.argsort()

#3 shap value 
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.summary_plot(shap_values, X_test)