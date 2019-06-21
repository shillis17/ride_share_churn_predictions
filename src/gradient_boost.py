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
from sklearn.metrics import confusion_matrix as cm
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import make_scorer
import numpy as np
import matplotlib.pyplot as plt

def prec(y_test, pred):
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    precision = tp/(tp+fp)
    return round(precision*100,2)

    return precision

def rec(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    recall = tp/(tp+fn)
    return round(recall*100,2)

def acc(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp+tn)/(tn+fn+fp+tp)
    return round(accuracy*100,2)
    
acc_score = make_scorer(acc)
prec_score = make_scorer(prec)
recall_score = make_scorer(rec)
    
def grid_search(X_train,y_train,model):
    params = {'booster':['gbtree','gblinear','dart'],
              'learning_rate':[.1,.3,.5,.7,.9],
              'max_depth':[1,3,5,7,9],
              'n_estimators':[10,30,50,70,90]}
    gs = GridSearchCV(model,params,cv=3,verbose=1,n_jobs=-1,scoring=prec_score)
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
    plt.savefig('../img/gradient_boost_featureImp.png')

def confusion_matrix(y_test,pred):
    matrix = cm(y_test,pred)
    tn, fp, fn, tp = matrix.ravel()
    real_mat = np.array([[tp,fn],[fp,tn]])
    return real_mat

def plot_confusion_matrix(y_true, y_pred,
                          title=None,
                          cmap=plt.cm.Blues):
    classes = unique_labels(y_test)
    normalize=True
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(12,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('../img/gradient_boost_confusionmatrix.png')
    return ax

df = pd.read_csv('../data/clean_train.csv')

X = df.drop(['churn','Unnamed: 0'],axis = 1)
y = df['churn']

X_train, X_test, y_train, y_test = tts(
        X, y, test_size=0.33)

model = XGBClassifier(booster='gbtree',
                      learning_rate=0.5,
                      max_depth=3,
                      n_estimators=50,
                      nthread=-1,
                      gamma=0.1,
                      max_delta_step=3,
                      min_child_weight=5,
                      subsample=.9)
model.fit(X_train,y_train)

pred = model.predict(X_test)


#print(grid_search(X_train,y_train,model))
graph()
print(confusion_matrix(y_test,pred))
plot_confusion_matrix(y_test,pred)
print('Precision:',prec(y_test,pred))
print('Recall:',rec(y_test,pred))
print('Accuracy:',acc(y_test,pred))