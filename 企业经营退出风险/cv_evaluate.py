#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:23:42 2017

@author: zeroquest
"""

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report 
from sklearn.metrics import roc_auc_score

import xgboost as xgb

from dankit import clfScore, answer

X_train = pd.read_pickle('X_train_with_recruit_pickle')
X_answer = pd.read_pickle('X_answer_with_recruit_pickle')

y_train = X_train['TARGET']
X_ID = X_answer['EID']

X_train.drop(['EID','TARGET'], axis=1, inplace=True)
X_answer.drop(['EID'], axis=1, inplace=True)

X_train = X_train.astype('float64')

#X_train, X_test, y_train, y_test = train_test_split(
#        X_train, y_train, test_size=0.18, random_state=0)

skf = StratifiedKFold(n_splits=10)
X_train_skf = []
X_test_skf = []
y_train_skf = []
y_test_skf = []
train_indexes = []
test_indexes = []
X_train.index = range(len(X_train))
y_train.index = range(len(y_train))
for train_index, test_index in skf.split(X_train, y_train):
    train_indexes.append(train_index)
    test_indexes.append(test_index)

for i in range(10):
    X_train_skf.append(X_train.loc[train_indexes[i], :])
    X_test_skf.append(X_train.loc[test_indexes[i], :])
    y_train_skf.append(y_train[train_indexes[i]])
    y_test_skf.append(y_train.loc[test_indexes[i]])

#XGBOOST
parameters = {  'max_depth': 14, 
                'learning_rate': 0.1, 
                'n_estimators': 50, 
                'silent': True, 
                'objective': 'binary:logistic', 
                'nthread': -1, 
                'gamma': 0,
                'min_child_weight': 10, 
                'max_delta_step': 5, 
                'subsample': 0.85, 
                'colsample_bytree': 0.7, 
                'colsample_bylevel': 1, 
                'reg_alpha': 0, 
                'reg_lambda': 1, 
                'scale_pos_weight': 1, 
                'seed': 10, 
                'missing': None}
param_test = {'max_depth': [12, 13, 14]}

xlf = []
for i in range(10):
    xlf.append(xgb.XGBRegressor( max_depth=14, 
                            learning_rate=0.1, 
                            n_estimators=70, 
                            silent=True, 
                            objective='binary:logistic', 
                            nthread=-1, 
                            gamma=0,
                            min_child_weight=10, 
                            max_delta_step=5, 
                            subsample=0.85, 
                            colsample_bytree=0.7, 
                            colsample_bylevel=1, 
                            reg_alpha=0, 
                            reg_lambda=1, 
                            scale_pos_weight=1, 
                            seed=10, 
                            missing=None))
#clf = GridSearchCV(xlf, param_test, cv=10,
#                       scoring='roc_auc') 
#clf.fit(X_train, y_train) 

for i in range(10): 
    print("Fold %d is training..." % i)
    xlf[i].fit(X_train_skf[i], y_train_skf[i],
            eval_metric='auc',
            verbose = True,
            eval_set = [(X_test_skf[i], y_test_skf[i])],
            early_stopping_rounds=1000)
#    clfScore(xlf[i], X_test_skf[i], y_test_skf[i])
    
#pred = []
#for i in range(10):
#    pred.append(xlf[i].predict(X_answer))

#pred_mean = np.mean(pred, axis=0)

#y_true, y_pred = y_test, clf.predict(X_test)  
#print(classification_report(y_true, y_pred))

#pred = answer(xlf, X_answer, X_ID)