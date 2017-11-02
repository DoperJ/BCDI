#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 22:13:06 2017

@author: zeroquest
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from dankit import splitDate, compress, pca_ratio_curve, addColumnsPrefix

X_train = pd.read_pickle('X_train_from_entbase_pickle')
X_answer = pd.read_pickle('X_answer_from_entbase_pickle')
alter = pd.read_csv('2alter.csv')

X = pd.concat([X_train.drop('TARGET', axis=1), X_answer])
X_alter = pd.merge(X, alter, how='left', on='EID').loc[:, ['EID', 'ALTERNO']]
X_alter.fillna(0, inplace=True)

#alter['ALTBE'].fillna('0', inplace=True)
#l = np.array([x[:-2] if len(x) > 2 else x for x in alter['ALTBE']])
#alter.loc[np.where(l == 'null')[0],'ALTBE'] = '0'
#alter['ALTBE'] = np.array(
#        [float(x[:-2]) if len(x) > 2 else float(x) for x in alter['ALTBE']])

#alter['ALTAF'].fillna('0', inplace=True)
#l = np.array([x[:-2] if len(x) > 2 else x for x in alter['ALTAF']])
#alter.loc[np.where(l == 'null')[0],'ALTAF'] = '0'
#alter['ALTAF'] = np.array(
#        [float(x[:-2]) if len(x) > 2 else float(x) for x in alter['ALTAF']])

#ALTDATE
alter['alt_year'], alter['alt_month'] = splitDate(alter['ALTDATE'])
alter_year_dummies = pd.get_dummies(alter['alt_year'])
alter_month_dummies = pd.get_dummies(alter['alt_month'])
addColumnsPrefix(alter_year_dummies, 'alter_year')
addColumnsPrefix(alter_month_dummies, 'alter_month')

alter_date_dummies = alter_year_dummies.join(alter_month_dummies)
alter_date_dummies[['EID']] = alter[['EID']]
alter_date_dummies = alter_date_dummies.groupby('EID').sum()
alter_date_dummies.reset_index(inplace=True)

#ALTERNO
alterno_dummies = pd.get_dummies(X_alter['ALTERNO'])
alterno_dummies[['EID']] = X_alter[['EID']]
alterno_dummies = alterno_dummies.groupby('EID').sum()
addColumnsPrefix(alterno_dummies, 'alterno')
alterno_dummies.reset_index(inplace=True)


#add alterno ont-hot columns
#alterno_dummies = alterno_compressed

#add alter_year and alter_month one-hot columns


alter_dummies = pd.merge(alterno_dummies, alter_date_dummies, how='left', on='EID')

X_train = pd.merge(X_train, alter_dummies, how='left', on='EID')
X_answer = pd.merge(X_answer, alter_dummies, how='left', on='EID')
X_train.fillna(0, inplace=True)
X_answer.fillna(0, inplace=True)

X_train.to_pickle('X_train_with_alter_pickle')
X_answer.to_pickle('X_answer_with_alter_pickle')