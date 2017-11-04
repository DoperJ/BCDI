#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 22:38:02 2017

@author: zeroquest
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dankit import pca_ratio_curve, compress, isExist, addColumnsPrefix, splitDate

X_train = pd.read_pickle('X_train_with_project_pickle')
X_answer = pd.read_pickle('X_answer_with_project_pickle')
lawsuit = pd.read_csv('7lawsuit.csv')

lawsuit['lawsuit_number'] = 1
lawsuit['lawsuit_year'], lawsuit['lawsuit_month'] = splitDate(lawsuit['LAWDATE'])


lawsuit_year_dummies = pd.get_dummies(lawsuit['lawsuit_year'])
lawsuit_month_dummies = pd.get_dummies(lawsuit['lawsuit_month'])
addColumnsPrefix(lawsuit_year_dummies, 'lawsuit_year')
addColumnsPrefix(lawsuit_month_dummies, 'lawsuit_month')
#将年份与月份合并为一个DataFrame
lawsuit_date_dummies = lawsuit_year_dummies.join(lawsuit_month_dummies)
lawsuit_date_dummies[['EID']] = lawsuit[['EID']]
lawsuit_date_dummies = lawsuit_date_dummies.groupby('EID').sum()
lawsuit_date_dummies.reset_index(inplace=True)

lawsuit_number = lawsuit[['EID', 'lawsuit_number']].groupby('EID').sum()
lawsuit_number.reset_index(inplace=True)

lawsuit_date_and_number = pd.merge(lawsuit_number, lawsuit_date_dummies, on='EID')
X_train = pd.merge(X_train, lawsuit_date_and_number, how='left', on='EID')
X_answer = pd.merge(X_answer, lawsuit_date_and_number, how='left', on='EID')

X_train.fillna(0, inplace=True)
X_answer.fillna(0, inplace=True)

X_train.to_pickle('X_train_with_lawsuit_pickle')
X_answer.to_pickle('X_answer_with_lawsuit_pickle')