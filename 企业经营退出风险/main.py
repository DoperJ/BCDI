#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 23:21:36 2017

@author: zeroquest
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

#models
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

#
from dankit import clfScore, answer

entbase = pd.read_csv('1entbase.csv')
alter = pd.read_csv('2alter.csv')
branch = pd.read_csv('3branch.csv')
X = pd.read_csv('train.csv')
evaluation = pd.read_csv('evaluation_public.csv')

#RGYEAR
entbase['year_or_old'] = list([2000 if x <= 2000 else x for x in entbase['RGYEAR']])

#HY
hy_dummies = pd.get_dummies(entbase['HY'])
#variances = []
##pca = PCA(n_components=21)
##hy_compressed = pca.fit_transform(hy_dummies)
##for i in range(1,21):
##    variances.append(pca.explained_variance_[:i].sum())
#plt.plot(np.arange(1,21),variances)
#plt.xticks(np.arange(1,21))
#plt.xlabel('numbers of features to keep')
#plt.ylabel('ratio of information remains')
#plt.annotate('Point(%d,%.2f)' % (10,variances[9]), xy=(10, variances[9]),
#            xytext=(+10, +0.7), fontsize=15,
#            arrowprops=dict(arrowstyle="->"))
#plt.show()
pca = PCA(n_components=15)
hy_compressed = pca.fit_transform(hy_dummies)
hy_compressed_df = pd.DataFrame(hy_compressed, 
             columns=list(['hy'+str(x) for x in range(1,16)]))
entbase = entbase.join(hy_compressed_df)

#ZCZB
imp_nan = Imputer(missing_values='NaN', strategy='median', axis=0)
imp_nan.fit(entbase.loc[:,['ZCZB']])
entbase.loc[:,['ZCZB']] = imp_nan.transform(entbase.loc[:,['ZCZB']])
imp_0 = Imputer(missing_values=0, strategy='median', axis=0)
imp_0.fit(entbase.loc[:,['ZCZB']])
entbase.loc[:,['ZCZB']] = imp_0.transform(entbase.loc[:,['ZCZB']])
scaler = StandardScaler()
scaler.fit(entbase['ZCZB'])
entbase['ZCZB'] = scaler.transform(entbase['ZCZB'])

#ETYPE
etype_compressed = pd.get_dummies(entbase['ETYPE'])
etype_compressed_df = pd.DataFrame(np.array(etype_compressed), 
             columns=list(['etype'+str(x) for x in sorted(entbase['ETYPE'].unique())]))
entbase = entbase.join(etype_compressed_df)

#~id standard
entbase.fillna(0, inplace=True)

X_train = pd.merge(X, entbase, how='left', on='EID')
X_answer = pd.merge(evaluation, entbase, how='left', on='EID')
#g = sns.factorplot(x='RGYEAR', y='TARGET', data=wX_train, aspect=3)
#g.set_xticklabels(step=5)
#X_train['RGYEAR'].describe()

X_train.drop(['RGYEAR','HY','ETYPE'], axis=1, inplace=True)
X_answer.drop(['RGYEAR','HY','ETYPE'], axis=1, inplace=True)

X_train.to_pickle('X_train_from_entbase_pickle')
X_answer.to_pickle('X_answer_from_entbase_pickle')