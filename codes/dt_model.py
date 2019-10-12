# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:06:24 2019

@author: ycai
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os
import random


def train_model(train_df, max_dep):
    Y = train_df['label'].tolist()
    X = train_df.drop(columns = ['label'])    
    model = DecisionTreeClassifier(criterion = 'entropy', max_depth = max_dep)
    model.fit(X, Y)
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
    if len(correct_classified) > sample_size:    
        temp = random.sample(correct_classified, sample_size)    
        return temp
    else:
        return correct_classified
    
def dt_model(dataset, max_dep, sample_size):
    os.chdir('/home/cai7/data/' + dataset)
    train_df = pd.read_pickle('{}_train_df.pkl'.format(dataset))
    test_df = pd.read_pickle('{}_test_df.pkl'.format(dataset))
    test_df = test_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    
    model = train_model(train_df, max_dep)
    accuracy, correct_classified = test_performance(model, test_df)
    samples = select_samples(correct_classified, sample_size)
    sample_df = test_df.iloc[samples, :]
    sample_df = sample_df.sort_index()
    
    return model, accuracy, sample_df