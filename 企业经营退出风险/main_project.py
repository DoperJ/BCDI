#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:27:06 2017

@author: zeroquest
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dankit import pca_ratio_curve, compress, isExist, addColumnsPrefix, splitDate
X_train = pd.read_pickle('X_train_with_right_pickle')
X_answer = pd.read_pickle('X_answer_with_right_pickle')
project = pd.read_csv('6project.csv')

project.rename(columns={'IFHOME':'project_at_home_number'}, inplace=True)
project['project_number'] = 1
X = pd.concat([X_train.drop('TARGET', axis=1), X_answer])
X_project = pd.merge(X, project, how='left', on='EID').loc[:,
                                                            ['EID',
                                                             'DJDATE',
                                                             'project_number',
                                                             'project_at_home_number']]
X_project.fillna(0, inplace=True)
X_project_numbers = X_project[['EID',
                               'project_number',
                               'project_at_home_number']].groupby('EID').sum()
X_project_numbers.reset_index(inplace=True)

#X_train = pd.merge(X_train, X_project_numbers, how='left', on='EID')
#X_answer = pd.merge(X_answer, X_project_numbers, how='left', on='EID')

#ASKDATE --拆分为年份跟日期两个属性
project['project_year'], project['project_month'] = splitDate(project['DJDATE'])
project_year_dummies = pd.get_dummies(project['project_year'])
project_month_dummies = pd.get_dummies(project['project_month'])
addColumnsPrefix(project_year_dummies, 'project_year')
addColumnsPrefix(project_month_dummies, 'project_month')
#将年份与月份合并为一个DataFrame
project_date_dummies = project_year_dummies.join(project_month_dummies)
project_date_dummies[['EID']] = project[['EID']]
project_date_dummies = project_date_dummies.groupby('EID').sum()
project_date_dummies.reset_index(inplace=True)

X_train = pd.merge(X_train, project_date_dummies, how='left', on='EID')
X_answer = pd.merge(X_answer, project_date_dummies, how='left', on='EID')

X_train.fillna(0, inplace=True)
X_answer.fillna(0, inplace=True)

X_train.to_pickle('X_train_with_project_pickle')
X_answer.to_pickle('X_answer_with_project_pickle')