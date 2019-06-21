#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:16:32 2019

@author: seth
"""
from sklearn.utils.multiclass import unique_labels
import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer

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

def graph():
    imp = model.feature_importances_
    items = []
    for i in(zip(X_test.columns,imp)):
        items.append(i)
        items.sort(key=lambda x:x[1],reverse=True)
        label = []
        value = []
        for i in range (len(items)):
            label.append(items[i][0])
            value.append(items[i][1])
    fig,ax = plt.subplots(figsize=(12,10))
    fig.tight_layout(pad=10)
    plt.xticks(rotation=90)
    plt.bar(label,value)
    plt.title("Feature importances")
    plt.xlim([-1, X_test.shape[1]])
    plt.xticks(rotation=45)
    plt.savefig('../img/final_model_featureImp.png',dpi=500)

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
    # We want to show all ticks...pred = model.predict(X_test)
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
    plt.savefig('../img/final_model_confusionmatrix.png')
    return ax

acc_score = make_scorer(acc)
prec_score = make_scorer(prec)
recall_score = make_scorer(rec)

data_train = pd.read_csv("../data/clean_train.csv", index_col="Unnamed: 0")
data_holdout = pd.read_csv("../data/clean_holdout.csv", index_col="Unnamed: 0")
y_train = data_train['churn']
X_train = data_train.drop(['churn'],axis=1)
y_test = data_holdout['churn']
X_test = data_holdout.drop(['churn'],axis=1)

model = XGBClassifier(booster='gbtree',
                      learning_rate=0.7,
                      max_depth=3,
                      n_estimators=30,
                      nthread=-1,
                      gamma=0.7,
                      max_delta_step=3,
                      min_child_weight=5,
                      subsample=1)
model.fit(X_train,y_train)

pred = model.predict(X_test)

graph()
plot_confusion_matrix(y_test,pred)
print('Precision:',prec(y_test,pred))
print('Recall:',rec(y_test,pred))
print('Accuracy:',acc(y_test,pred))
rec_list = []

'''
for i in range(0,10000):
    pred = model.predict(X_test)
    recall = rec(y_test,pred)
    rec_list.append(recall)
    print(np.average(rec_list))
'''