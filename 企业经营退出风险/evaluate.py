#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 23:13:18 2017

@author: zeroquest
"""

import pandas as pd


from sklearn.cross_validation import train_test_split

import xgboost as xgb

from dankit import clfScore, answer

X_train = pd.read_pickle('X_train_with_branch_pickle')
X_answer = pd.read_pickle('X_answer_with_branch_pickle')

y_train = X_train['TARGET']
X_ID = X_answer['EID']

X_train.drop(['EID','TARGET'], axis=1, inplace=True)
X_answer.drop(['EID'], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.3, random_state=0)

##Random Forest
#forest = RandomForestClassifier(criterion='entropy',
#                                n_estimators=20, 
#                                random_state=0)
#forest.fit(X_train, y_train)
#clfScore(forest, X_test, y_test)

#XGBOOST(Average 0.85)
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')
xlf = xgb.XGBRegressor(max_depth=10, 
                        learning_rate=0.1, 
                        n_estimators=50, 
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
                        missing=None)
xlf.fit(X_train, y_train,
        eval_metric='auc',
        verbose = True,
        eval_set = [(X_test, y_test)],
        early_stopping_rounds=1000)
clfScore(xlf, X_test, y_test)
#pred = answer(xlf, X_answer, X_ID)