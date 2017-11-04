#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:05:15 2017

@author: zeroquest
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from dankit import compress, pca_ratio_curve, addColumnsPrefix, isExist

X_train = pd.read_pickle('X_train_with_alter_pickle')
X_answer = pd.read_pickle('X_answer_with_alter_pickle')
branch = pd.read_csv('3branch.csv')


branch['branch_number'] = 1
#branch['branch_end_number'] = isExist(branch['B_ENDYEAR'])

branch_number = branch[['EID','IFHOME',
                        'branch_number']].groupby('EID').sum()
branch_number.reset_index(inplace=True)


#B_REYEAR 
branch['year_or_old_endyear'] = list([2000 if x <= 2000 else x for x in branch['B_ENDYEAR']])
branch['year_or_old_reyear'] = list([1990 if x <= 1990 else x for x in branch['B_REYEAR']])
re_year_dummies = pd.get_dummies(branch['year_or_old_reyear'])
addColumnsPrefix(re_year_dummies, 'branch_re_year')
re_year_dummies['EID'] = branch['EID']
re_year_dummies = re_year_dummies.groupby('EID').sum()
#B_ENDYEAR
end_year_dummies = pd.get_dummies(branch['year_or_old_endyear'])
addColumnsPrefix(end_year_dummies, 'branch_end_year')
end_year_dummies['EID'] = branch['EID']
end_year_dummies = end_year_dummies.groupby('EID').sum()


#绘制pca压缩后保留数据百分比的曲线，确定压缩维度
#pca_ratio_curve(re_year_dummies, 20, 4)
re_year_compressed_df = compress(re_year_dummies, 4, 're_year')
re_year_dummies.reset_index(inplace=True)
re_year_compressed_df['EID'] = re_year_dummies['EID']
#绘制pca压缩后保留数据百分比的曲线，确定压缩维度
#pca_ratio_curve(end_year_dummies, 15, 8)
end_year_compressed_df = compress(end_year_dummies, 8, 'end_year')
end_year_dummies.reset_index(inplace=True)
end_year_compressed_df['EID'] = end_year_dummies['EID']

#将re_year、end_year属性集与branch_number属性合并为一个DataFrame--branch
branch_year = pd.merge(re_year_dummies, end_year_dummies, how='left', on='EID')
branch = pd.merge(branch_number, re_year_dummies, how='left', on='EID')

#数据分别并入训练集和测试集
X_train = pd.merge(X_train, branch, how='left', on='EID')
X_answer = pd.merge(X_answer, branch, how='left', on='EID')
X_train.fillna(0, inplace=True)
X_answer.fillna(0, inplace=True)

X_train.to_pickle('X_train_with_branch_pickle')
X_answer.to_pickle('X_answer_with_branch_pickle')