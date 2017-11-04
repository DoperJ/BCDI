#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:56:21 2017

@author: zeroquest
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dankit import isExist, addColumnsPrefix, splitDate, getDateDummies

X_train = pd.read_pickle('X_train_with_lawsuit_pickle')
X_answer = pd.read_pickle('X_answer_with_lawsuit_pickle')
breakfaith = pd.read_csv('8breakfaith.csv')


breakfaith['breakfaith_number'] = 1
breakfaith['breakfaith_end_number'] = isExist(breakfaith['SXENDDATE'])
breakfaith['breakfaith_year'], breakfaith['breakfaith_month'] = splitDate(breakfaith['FBDATE'], '/')

#FBDATE
breakfaith_date_dummies = getDateDummies(breakfaith['breakfaith_year'],
                                         breakfaith['breakfaith_month'],
                                         'breakfaith_year', 'breakfaith_month',
                                         breakfaith['EID'])

#SXENDDATE
mask = [False if str(n) == "nan" else True for n in breakfaith['SXENDDATE']]
breakfaith_end_date = breakfaith.loc[mask, ['EID', 'SXENDDATE']]
breakfaith_end_date['breakfaith_end_year'], breakfaith_end_date['breakfaith_end_month'] = splitDate(breakfaith_end_date['SXENDDATE'], '/')
breakfaith_end_date_dummies = getDateDummies(breakfaith_end_date['breakfaith_end_year'],
                                         breakfaith_end_date['breakfaith_end_month'],
                                         'breakfaith_end_year', 'breakfaith_end_month',
                                         breakfaith_end_date['EID'])

breakfaith_numbers = breakfaith[['EID',
                                 'breakfaith_number',
                                 'breakfaith_end_number']].groupby('EID').sum()
breakfaith_numbers.reset_index(inplace=True)

breakfaith_date_and_numbers = pd.merge(breakfaith_numbers,
                                       breakfaith_date_dummies,
                                       how='left', on='EID')
#breakfaith_date_and_numbers = pd.merge(breakfaith_date_and_numbers,
#                                       breakfaith_end_date_dummies,
#                                       how='left', on='EID')
X_train = pd.merge(X_train, breakfaith_date_and_numbers, how='left', on='EID')
X_answer = pd.merge(X_answer, breakfaith_date_and_numbers, how='left', on='EID')

X_train.fillna(0, inplace=True)
X_answer.fillna(0, inplace=True)

X_train.to_pickle('X_train_with_breakfaith_pickle')
X_answer.to_pickle('X_answer_with_breakfaith_pickle')