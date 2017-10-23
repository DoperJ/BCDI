#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:13:06 2017

@author: zeroquest
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

X_train = pd.read_pickle('X_train_from_entbase_pickle')
X_answer = pd.read_pickle('X_answer_from_entbase_pickle')
alter = pd.read_csv('2alter.csv')

alter['ALTBE'].fillna('0', inplace=True)
l = np.array([x[:-2] if len(x) > 2 else x for x in alter['ALTBE']])
alter.loc[np.where(l == 'null')[0],'ALTBE'] = '0'
alter['ALTBE'] = np.array(
        [float(x[:-2]) if len(x) > 2 else float(x) for x in alter['ALTBE']])


alter['ALTAF'].fillna('0', inplace=True)
l = np.array([x[:-2] if len(x) > 2 else x for x in alter['ALTAF']])
alter.loc[np.where(l == 'null')[0],'ALTAF'] = '0'
alter['ALTAF'] = np.array(
        [float(x[:-2]) if len(x) > 2 else float(x) for x in alter['ALTAF']])

#ALTDATE
alter_date = np.array([x.split('-') for x in alter['ALTDATE']])
alter['alt_year'], alter['alt_month'] = alter_date[:, 0], alter_date[:, 1]
alter_year_dummies = pd.get_dummies(alter['alt_year'])
alter_month_dummies = pd.get_dummies(alter['alt_month'])
alter_year_dummies.columns=list(['alt_year'+str(x) for x in alter_year_dummies.columns])
alter_month_dummies.columns=list(['alt_month'+str(x) for x in alter_month_dummies.columns])

alter_date_dummies = alter_year_dummies.join(alter_month_dummies)
alter_date_dummies[['EID']] = alter[['EID']]
alter_date_dummies = alter_date_dummies.groupby('EID').sum()
alter_date_dummies.reset_index(inplace=True)

#ALTERNO
alterno_dummies = pd.get_dummies(alter['ALTERNO'])
alterno_dummies[['EID']] = alter[['EID']]
alterno_dummies = alterno_dummies.groupby('EID').sum()
alterno_dummies.reset_index(inplace=True)

#variances = []
#pca = PCA(n_components=10)
#alterno_compressed = pca.fit_transform(alterno_dummies.iloc[:,1:])
#for i in range(1,11):
#    variances.append(pca.explained_variance_ratio_[:i].sum())
#plt.plot(np.arange(1,11),variances)
#plt.xticks(np.arange(1,11))
#plt.xlabel('numbers of features to keep')
#plt.ylabel('ratio of information remains')
#plt.annotate('Point(%d,%.2f)' % (9,variances[8]), xy=(9, variances[8]),
#            xytext=(+8, +0.7), fontsize=15,
#            arrowprops=dict(arrowstyle="->"))
#plt.show()
pca = PCA(n_components=10)
alterno_compressed = pca.fit_transform(alterno_dummies.iloc[:, 1:-2])
alterno_compressed_df = pd.DataFrame(alterno_compressed, 
             columns=list(['alterno'+str(x) for x in range(1,11)]))
alterno_compressed_df['EID'] = alterno_dummies['EID']

#add alterno ont-hot columns
#alterno_dummies = alterno_compressed

#add alter_year and alter_month one-hot columns


alter_dummies = pd.merge(alter_date_dummies, alterno_compressed_df, on='EID')
#alter = pd.merge(alter, alter_date_dummies, on='EID')
#alter.drop(['ALTERNO', 'ALTDATE', 'ALTBE',
#            'ALTAF', 'alt_year', 'alt_month'], axis=1, inplace=True)
#X_answer.drop(['ALTERNO', 'ALTDATE','ALTBE', 'ALTAF'], axis=1, inplace=True)
X_train = pd.merge(X_train, alter_dummies, how='left', on='EID')
X_answer = pd.merge(X_answer, alter_dummies, how='left', on='EID')
X_train.fillna(0, inplace=True)
X_answer.fillna(0, inplace=True)

X_train.to_pickle('X_train_with_alter_pickle')
X_answer.to_pickle('X_answer_with_alter_pickle')