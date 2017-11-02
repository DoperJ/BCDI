#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:38:05 2017

@author: zeroquest
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from dankit import compress, pca_ratio_curve, addColumnsPrefix, isExist

X_train = pd.read_pickle('X_train_with_branch_pickle')
X_answer = pd.read_pickle('X_answer_with_branch_pickle')
invest = pd.read_csv('4invest.csv')

invest.rename(columns={'IFHOME':'invest_at_home_number'}, inplace=True)
#拆分为两个表，分别表示投资和被投资数据
invested = invest[['BTEID', 'BTBL']]
invest = invest[['EID', 'invest_at_home_number', 'BTBL', 'BTYEAR', 'BTENDYEAR']]
#自定义属性，invest_others表示是否对外投资，
invest['invest_others'] = 1
#invest_to_end表示投资是否失败
invest['invest_to_end'] = isExist(invest['BTENDYEAR'])
#is_invested表示企业是否被投资
invested['is_invested'] = 1
X = pd.concat([X_train.drop('TARGET', axis=1), X_answer])

#==============================================================================
#                       X_invest --投资部分
#==============================================================================
X_invest = pd.merge(X, invest, how='left', on='EID').loc[:, 
                                                        ['EID', 
                                                         'invest_others',
                                                         'invest_at_home_number',
                                                         'invest_to_end',
                                                         'BTBL']]

X_invest.fillna(0, inplace=True)
X_invest_numbers = X_invest[['EID', 
                             'invest_others', 
                             'invest_at_home_number', 
                             'invest_to_end',
                             'BTBL']].groupby('EID').sum()
X_invest_numbers.reset_index(inplace=True)

X_train = pd.merge(X_train, X_invest_numbers, on='EID')
X_answer = pd.merge(X_answer, X_invest_numbers, on='EID')

#BTYEAR
invest_year_dummies = pd.get_dummies(invest['BTYEAR'])
#addColumnsPrefix(invest_year_dummies, 'invest_year')
invest_year_dummies['EID'] = invest['EID']

invest_year = invest_year_dummies.groupby('EID').sum()
#pca_ratio_curve(invest_year, 40, 25)
invest_year = compress(invest_year, 18, 'invest_year', keepEID=True)

#X_train = pd.merge(X_train, invest_year, how='left', on='EID')
#X_answer = pd.merge(X_answer, invest_year, how='left', on='EID')


#BTENDYEAR
invest_year_end = invest.loc[~np.isnan(invest['BTENDYEAR']), ['EID', 'BTENDYEAR']]
invest_year_end_dummies = pd.get_dummies(invest_year_end['BTENDYEAR'])
addColumnsPrefix(invest_year_end_dummies, 'invest_year')
invest_year_end_dummies['EID'] = invest_year_end['EID']
invest_year_end = invest_year_end_dummies.groupby('EID').sum()
invest_year_end.reset_index(inplace=True)

X_train = pd.merge(X_train, invest_year_end, how='left', on='EID')
X_answer = pd.merge(X_answer, invest_year_end, how='left', on='EID')

#==============================================================================
#                           X_invested --被投资部分
#==============================================================================
invested.rename(columns={'BTEID':'EID'}, inplace=True)
X_invested = pd.merge(X, invested, how='left', on='EID').loc[:, 
                                                        ['EID',
                                                         'BTBL',
                                                         'is_invested']]

X_invested.fillna(0, inplace=True)
X_invested_numbers = X_invested[['EID', 'BTBL', 'is_invested']].groupby('EID').sum()
X_invested_numbers.reset_index(inplace=True)


X_train = pd.merge(X_train, X_invested_numbers, how='left', on='EID')
X_answer = pd.merge(X_answer, X_invested_numbers, how='left', on='EID')

X_train.fillna(0, inplace=True)
X_answer.fillna(0, inplace=True)

X_train.to_pickle('X_train_with_invest_pickle')
X_answer.to_pickle('X_answer_with_invest_pickle')