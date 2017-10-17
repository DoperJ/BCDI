#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 22:06:09 2017

@author: zeroquest
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

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
    return prediction