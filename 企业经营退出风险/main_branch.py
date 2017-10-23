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

from dankit import compress, pca_ratio_curve

X_train = pd.read_pickle('X_train_with_alter_pickle')
X_answer = pd.read_pickle('X_answer_with_alter_pickle')
branch = pd.read_csv('3branch.csv')

branch['branch_number'] = 1
#branch['branch_end_number'] = (~branch['B_ENDYEAR'].map(np.isnan)).astype(int)

branch_number = branch[['EID','IFHOME', 'branch_number']].groupby('EID').sum()
branch_number.reset_index(inplace=True)


#B_REYEAR 
branch['year_or_old_endyear'] = list([2000 if x <= 2000 else x for x in branch['B_ENDYEAR']])
branch['year_or_old_reyear'] = list([1990 if x <= 1990 else x for x in branch['B_REYEAR']])
re_year_dummies = pd.get_dummies(branch['year_or_old_reyear'])
re_year_dummies['EID'] = branch['EID']
re_year_dummies = re_year_dummies.groupby('EID').sum()
#variances = []
#pca = PCA(n_components=20)
#re_year_compressed = pca.fit_transform(re_year_dummies)
#for i in range(1,21):
#    variances.append(pca.explained_variance_ratio_[:i].sum())
#plt.plot(np.arange(1,21),variances)
#plt.xticks(np.arange(1,21))
#plt.xlabel('numbers of features to keep')
#plt.ylabel('ratio of information remains')
#plt.annotate('Point(%d,%.2f)' % (4,variances[3]), xy=(4, variances[3]),
#            xytext=(+4, +0.9), fontsize=15,
#            arrowprops=dict(arrowstyle="->"))
#plt.show()
pca = PCA(n_components=4)
re_year_compressed = pca.fit_transform(re_year_dummies)
re_year_compressed_df = pd.DataFrame(re_year_compressed, 
             columns=list(['re_year'+str(x) for x in range(1,5)]))
re_year_dummies.reset_index(inplace=True)
re_year_compressed_df['EID'] = re_year_dummies['EID']

#B_ENDYEAR
end_year_dummies = pd.get_dummies(branch['year_or_old_endyear'])
end_year_dummies['EID'] = branch['EID']
end_year_dummies = end_year_dummies.groupby('EID').sum()

#绘制pca压缩后保留数据百分比的曲线，确定压缩维度
pca_ratio_curve(end_year_dummies, 15, 8)

#压缩end_year的one-hot编码至8维
#pca = PCA(n_components=8)
#end_year_compressed = pca.fit_transform(end_year_dummies)
#end_year_compressed_df = pd.DataFrame(end_year_compressed, 
#             columns=list(['end_year'+str(x) for x in range(1,9)]))
#end_year_dummies.reset_index(inplace=True)
#end_year_compressed_df['EID'] = end_year_dummies['EID']
#
#branch_year = pd.merge(re_year_compressed_df, end_year_compressed_df, how='left', on='EID')
#branch = pd.merge(branch_number, branch_year, how='left', on='EID')

#branch.drop([''], axis=1, inplace=True)

#模型合并
#X_train = pd.merge(X_train, branch, how='left', on='EID')
#X_answer = pd.merge(X_answer, branch, how='left', on='EID')
#X_train.fillna(0, inplace=True)
#X_answer.fillna(0, inplace=True)

#尝试使用pca，能降至一维且保留绝大部分数据但是效果不好
#X = pd.concat([X_train.drop(['TARGET'], axis=1), X_answer], ignore_index=True)
#
#X_compressed = compress(X.drop('EID', axis=1), n=25, prefix='feature')
#X_compressed['EID'] = X['EID']
#
#X_train = pd.merge(X_compressed[X['EID'].isin(X_train['EID'])],
#                                X_train[['EID', 'TARGET']], on='EID')
#X_train.reset_index(range(0, X_answer.shape[0]), drop=True, inplace=True)
#
#X_answer = X_compressed[X['EID'].isin(X_answer['EID'])]
#X_answer.reset_index(range(0, X_answer.shape[0]), drop=True, inplace=True)
#X_train.to_pickle('X_train_with_branch_pickle')
#X_answer.to_pickle('X_answer_with_branch_pickle')