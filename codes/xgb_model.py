# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:59:06 2019

@author: ycai
"""

import xgboost
import os
import pandas as pd
import random


def train_model(train_df, parameters, nclasses):
    Y = train_df['label'].tolist()
    X = train_df.drop(columns = ['label'])
    max_dep = parameters['max_dep']
    ntrees = parameters['ntrees']
    if nclasses == 2:
        objective = 'binary:logistic'
        model = xgboost.XGBClassifier(n_estimators = ntrees,
                                      objective = objective,                            
                                      max_depth = max_dep,   
                                      n_jobs = -1)
    else:
        objective = 'multi:softmax'
        nclasses = nclasses    
        model = xgboost.XGBClassifier(n_estimators = ntrees,
                                      objective = objective,                            
                                      max_depth = max_dep,    
                                      num_class = nclasses,
                                      n_jobs = -1)
    model.fit(X, Y)
    model.score(X,Y)
    return model
    
def test_performance(model, test_df):
    truey = test_df['label']
    X = test_df.drop(columns = ['label'])
    prediction = model.predict(X)
    temp = pd.DataFrame()
    temp['true'] = truey
    temp['pred'] = prediction
    correct = 0
    correct_classified = []
    for i in range(temp.shape[0]):
        if temp.iloc[i]['true'] == temp.iloc[i]['pred']:
            correct = correct + 1
            correct_classified.append(i)
    return (correct / temp.shape[0]), correct_classified
    
def select_samples(correct_classified, sample_size):
    temp = random.sample(correct_classified, sample_size)    
    return temp
    
def xgb_model(dataset, parameters, nclasses, sample_size):
    os.chdir('/home/cai7/data/' + dataset)
    train_df = pd.read_pickle('{}_train_df.pkl'.format(dataset))
    test_df = pd.read_pickle('{}_test_df.pkl'.format(dataset))
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    
    model = train_model(train_df, parameters, nclasses)
    accuracy, correct_classified = test_performance(model, test_df)
    samples = select_samples(correct_classified, sample_size)
    sample_df = test_df.iloc[samples, :]
    sample_df = sample_df.sort_index()
    
    return model, accuracy, sample_df
    
    
    
    
    