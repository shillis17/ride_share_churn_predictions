#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 10:57:54 2019

@author: seth
"""

from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

def accuracy(pred,y_test):
    corr = [a for a in range (len(pred)) if pred[a] == y_test[a]]
    acc = len(corr)/len(pred)
    return(round(acc*100,2))
    
def grid_search(X_train,y_train,model):
    params = {'gamma':[.1,.3,.5,.7,.9],
              'min_child_weight':[1,3,5,7,9],
              'max_delta_step':[1,3,5,7,9],
              'subsample':[.1,.3,.5,.7,.9]}
    gs = GridSearchCV(model,params,cv=3,verbose=1,n_jobs=-1)
    gs.fit(X_train,y_train)
    return(gs.best_params_)
    
def graph():
    imp = model.feature_importances_
    items = []
    for i in(zip(X.columns,imp)):
        items.append(i)
        items.sort(key=lambda x:x[1],reverse=True)
        label = []
        value = []
        for i in range (len(items)):
            label.append(items[i][0])
            value.append(items[i][1])    
    fig,ax = plt.subplots(figsize=(12,10))
    plt.bar(label,value)
    plt.title("Feature importances")
    plt.xlim([-1, X.shape[1]])
    plt.xticks(rotation=90)

df = pd.read_csv('../data/clean_train.csv')

X = df.drop(['churn','Unnamed: 0'],axis = 1)
y = df['churn']

X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.33)

model = XGBClassifier(booster='gbtree',
                      learning_rate=0.3,
                      max_depth=5,
                      n_estimators=50,
                      nthread=-1,
                      gamma=0.1,
                      max_delta_step=3,
                      min_child_weight=1,
                      subsample=1)
model.fit(X_train,y_train)

pred = model.predict(X_test)

print('Accuracy:',accuracy(list(pred),list(y_test)))
#print(grid_search(X_train,y_train,model))
graph()