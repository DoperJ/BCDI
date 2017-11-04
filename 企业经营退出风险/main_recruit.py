#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 00:33:00 2017

@author: zeroquest
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer


from dankit import pca_ratio_curve, compress, isExist, addColumnsPrefix, splitDate

X_train = pd.read_pickle('X_train_with_breakfaith_pickle')
X_answer = pd.read_pickle('X_answer_with_breakfaith_pickle')
recruit = pd.read_csv('9recruit.csv')

recruit['recruit_times'] = 1
recruit['recruit_year'], recruit['recruit_month'] = splitDate(recruit['RECDATE'])

recruit_year_dummies = pd.get_dummies(recruit['recruit_year'])
recruit_month_dummies = pd.get_dummies(recruit['recruit_month'])
addColumnsPrefix(recruit_year_dummies, 'recruit_year')
addColumnsPrefix(recruit_month_dummies, 'recruit_month')
#将年份与月份合并为一个DataFrame
recruit_date_dummies = recruit_year_dummies.join(recruit_month_dummies)
recruit_date_dummies[['EID']] = recruit[['EID']]
recruit_date_dummies = recruit_date_dummies.groupby('EID').sum()
recruit_date_dummies.reset_index(inplace=True)


recruit_website_dummies = pd.get_dummies(recruit['WZCODE'])
addColumnsPrefix(recruit_website_dummies, 'recruit_website_')
recruit_website_dummies[['EID']] = recruit[['EID']]
recruit_website_dummies = recruit_website_dummies.groupby('EID').sum()
recruit_website_dummies.reset_index(inplace=True)

imp_nan = Imputer(missing_values='NaN', strategy='median', axis=0)
imp_nan.fit(recruit.loc[:,['RECRNUM']])
recruit.loc[:,['RECRNUM']] = imp_nan.transform(recruit.loc[:,['RECRNUM']])

recruit_numbers = recruit[['EID', 'recruit_times', 'RECRNUM']].groupby('EID').sum()
recruit_numbers.reset_index(inplace=True)

recruit_date_and_numbers = pd.merge(recruit_date_dummies,
                                    recruit_numbers, 
                                    how='left', on='EID')

#合并
recruit_date_website_and_numbers = pd.merge(recruit_date_and_numbers,
                                            recruit_website_dummies,
                                            how='left', on='EID')
X_train = pd.merge(X_train, recruit_date_website_and_numbers, how='left', on='EID')
X_answer = pd.merge(X_answer, recruit_date_website_and_numbers, how='left', on='EID')


X_train.fillna(0, inplace=True)
X_answer.fillna(0, inplace=True)

X_train.to_pickle('X_train_with_recruit_pickle')
X_answer.to_pickle('X_answer_with_recruit_pickle')