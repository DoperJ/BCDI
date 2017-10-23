#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:38:05 2017

@author: zeroquest
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

X_train = pd.read_pickle('X_train_with_branch_pickle')
X_answer = pd.read_pickle('X_answer_with_branch_pickle')
invest = pd.read_csv('4invest.csv')


