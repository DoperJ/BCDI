#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:18:12 2017

@author: zeroquest
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dankit import pca_ratio_curve, compress, isExist, addColumnsPrefix, splitDate

X_train = pd.read_pickle('X_train_with_invest_pickle')
X_answer = pd.read_pickle('X_answer_with_invest_pickle')
right = pd.read_csv('5right.csv')

X = pd.concat([X_train.drop('TARGET', axis=1), X_answer])

X_right = pd.merge(X, right, how='left', on='EID').loc[:, 
                                                        ['EID', 'RIGHTTYPE',
                                                         'TYPECODE', 'ASKDATE',
                                                         'FBDATE']]

#RIGHTTYPE
right_type_dummies = pd.get_dummies(right['RIGHTTYPE'].fillna(0))
right_type_dummies['EID'] = right['EID']
right_type_dummies = right_type_dummies.groupby('EID').sum()
addColumnsPrefix(right_type_dummies, 'right_type')
right_type_dummies.reset_index(inplace=True)
#pca_ratio_curve(right_type_dummies, 7, 4)

#新定义两个属性 --right_get描述的是获得权利的数目
right_get = X_right[['EID', 'FBDATE']]
right_get.loc[:, 'right_get'] = isExist(X_right['FBDATE'])
right_get = right_get.groupby('EID').sum()
#right_applied描述的是权利申请的数目
right_applied = X_right[['EID', 'RIGHTTYPE']]
right_applied.loc[:, 'right_applied'] = isExist(X_right['RIGHTTYPE'])
right_applied = right_applied.groupby('EID').sum()
#取消键'EID',方便使用merge
right_get.reset_index(inplace=True)
right_applied.reset_index(inplace=True)

#ASKDATE --拆分为年份跟日期两个属性
right['right_year'], right['right_month'] = splitDate(right['ASKDATE'])
right_year_dummies = pd.get_dummies(right['right_year'])
right_month_dummies = pd.get_dummies(right['right_month'])
addColumnsPrefix(right_year_dummies, 'right_year')
addColumnsPrefix(right_month_dummies, 'right_month')
#将年份与月份合并为一个DataFrame
right_date_dummies = right_year_dummies.join(right_month_dummies)
right_date_dummies[['EID']] = right[['EID']]
right_date_dummies = right_date_dummies.groupby('EID').sum()
right_date_dummies.reset_index(inplace=True)

#FBDATE --拆分为年份和日期两个属性
mask = [False if str(n) == "nan" else True for n in right['FBDATE']]
right_year_get, right_month_get = splitDate(right.loc[mask, 'FBDATE'])
right_year_get_df = pd.DataFrame(columns=['EID','right_year_get'])
right_year_get_df['EID'] = right.loc[mask, 'EID']
right_year_get_df['right_year_get'] = right_year_get
right_month_get_df = pd.DataFrame(columns=['EID','right_month_get'])
right_month_get_df['EID'] = right.loc[mask, 'EID']
right_month_get_df['right_month_get'] = right_month_get
#将right_year_get转换为one-hot编码
right_year_get_dummies = pd.get_dummies(right_year_get_df.drop('EID', axis=1))
right_year_get_dummies['EID'] = right_year_get_df['EID']
right_year_get_dummies = right_year_get_dummies.groupby('EID').sum()
#将right_month_get转换为one-hot编码
right_month_get_dummies = pd.get_dummies(right_month_get_df.drop('EID', axis=1))
right_month_get_dummies['EID'] = right_month_get_df['EID']
right_month_get_dummies = right_month_get_dummies.groupby('EID').sum()
#将两者合并为一个DataFrame
right_date_get_dummies = right_year_get_dummies.join(right_month_get_dummies)
right_date_get_dummies.reset_index(inplace=True)


scaler = StandardScaler()
scaler.fit(right_get['right_get'])
right_get['right_get'] = scaler.transform(right_get['right_get'])
scaler = StandardScaler()
scaler.fit(right_applied['right_applied'])
right_applied['right_applied'] = scaler.transform(right_applied['right_applied'])

#
right_compressed = pd.merge(right_get, right_type_dummies, how='left', on='EID')
right_compressed = pd.merge(right_compressed, right_applied, on='EID')
right_compressed = pd.merge(right_compressed, right_date_dummies, how='left', on='EID')
#right_compressed = pd.merge(right_compressed, right_date_get_dummies, how='left', on='EID')


right_compressed.drop(['RIGHTTYPE'], axis=1, inplace=True)
right_compressed.fillna(0, inplace=True)

X_train = pd.merge(X_train, right_compressed, how='left', on='EID')
X_answer = pd.merge(X_answer, right_compressed, how='left', on='EID')

X_train.to_pickle('X_train_with_right_pickle')
X_answer.to_pickle('X_answer_with_right_pickle')