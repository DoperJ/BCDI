#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 22:06:09 2017

@author: zeroquest
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def isExist(ser):
    mask = [False if str(n) == "nan" else True for n in ser]
    return np.array(mask, dtype=int)

def splitDate(ser, delimiter='-'):
    year_and_month = np.array([x.split(delimiter) for x in ser])
    return year_and_month[:, 0], year_and_month[:, 1]

def addColumnsPrefix(df, prefix):
    df.columns = list([prefix+str(x) for x in df.columns])

def getDateDummies(year, month, year_name, month_name, index):
    year_dummies = pd.get_dummies(year)
    month_dummies = pd.get_dummies(month)
    addColumnsPrefix(year_dummies, year_name)
    addColumnsPrefix(month_dummies, month_name)
    #将年份与月份合并为一个DataFrame
    date_dummies = year_dummies.join(month_dummies)
    date_dummies['EID'] = index
    date_dummies = date_dummies.groupby('EID').sum()
    date_dummies.reset_index(inplace=True)
    return date_dummies

def clfScore(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_pred = np.array([round(x) for x in y_pred])
    print('Accuracy: %.2f' % accuracy_score(y_pred, y_test))
    print(y_pred.sum())

def answer(clf, X_answer, X_ID):
    #make prediction array
    y_answer_prob = clf.predict(X_answer)
    y_answer = np.array([int(round(x)) for x in y_answer_prob])
    y_answer_prob = np.array([str(round(x,4)) for x in y_answer_prob])
    #transform into formated dataframe
    prediction = pd.DataFrame(columns=['EID','FORTARGET','PROB'])
    prediction['EID'] = X_ID
    prediction['FORTARGET'] = y_answer
    prediction['PROB'] = y_answer_prob
    prediction.set_index('EID', inplace=True)
    return prediction

def pca_ratio_curve(data, max_components, note_components=None):
    pca = PCA(n_components=max_components)
    data_compressed = pca.fit_transform(data)
    variances_sum = pca.explained_variance_ratio_.cumsum()
    plt.plot(np.arange(1,max_components+1),variances_sum)
    plt.xticks(np.arange(1,max_components+1))
    plt.xlabel('numbers of features to keep')
    plt.ylabel('ratio of information remains')
    if note_components:
        plt.annotate('Point(%d,%.2f)' % (note_components,
                     variances_sum[note_components-1]),
                    xy=(note_components, variances_sum[note_components-1]),
                    xytext=(note_components, variances_sum[note_components//2]),
                    fontsize=15,
                    arrowprops=dict(arrowstyle="->"))
    plt.show()
    return data_compressed

def compress(data, n, prefix, keepEID=False):
    pca = PCA(n_components=n)
    data_compressed = pca.fit_transform(data)
    data_compressed_df = pd.DataFrame(data_compressed, 
                 columns=list([prefix+str(x) for x in range(1,pca.n_components_+1)]))
    if keepEID:
        data_compressed_df['EID'] = data.index
    return data_compressed_df