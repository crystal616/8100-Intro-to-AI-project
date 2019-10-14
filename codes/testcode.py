# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 16:38:23 2019

@author: Ying
"""

import sys
sys.path.insert(0, '/home/cai7/RobustTrees')
import xgboost
import os
import pandas as pd
import random


def train_model(train_df, parameters, nclasses):
    Y = train_df['label'].tolist()
    X = train_df.drop(columns = ['label'])
    max_dep = parameters['max_depth']
    ntrees = parameters['n_estimators']
    eps = parameters['robust_eps']
    if nclasses == 2:
        objective = 'binary:logistic'
        model = xgboost.XGBClassifier(n_estimators = ntrees,
                                      objective = objective,
                                      max_depth = max_dep,
                                      robust_eps = eps,
                                      tree_method = "robust_exact",
                                      n_jobs = -1)
    else:
        objective = 'multi:softmax'
        nclasses = nclasses    
        model = xgboost.XGBClassifier(n_estimators = ntrees,
                                      objective = objective,                            
                                      max_depth = max_dep,    
                                      num_class = nclasses,
                                      robust_eps = eps,
                                      tree_method = "robust_exact",
                                      n_jobs = -1)
    model.fit(X, Y)
    model.score(X,Y)
    return model

dataset = 'breast_cancer'
nclasses = 2
parameters = {'max_depth':8, 'n_estimators':4, 'robust_eps':0.3}

os.chdir('/home/cai7/data/' + dataset)
train_df = pd.read_pickle('{}_train_df.pkl'.format(dataset))
test_df = pd.read_pickle('{}_test_df.pkl'.format(dataset))
test_df = test_df.reset_index(drop=True)
train_df = train_df.reset_index(drop=True)

model = train_model(train_df, parameters, nclasses)
accuracy, correct_classified = test_performance(model, test_df)
accuracy


os.chdir('/home/cai7/data/rxgb')
data = load_svmlight_file('breast_cancer_test.svm', n_features = 9, zero_based = False)

os.chdir('/home/cai7/RobustTrees/data')
data = load_svmlight_file('breast_cancer_scale0.test')