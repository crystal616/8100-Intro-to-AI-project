# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:04:33 2019

@author: ycai
"""

import os
os.chdir('/home/cai7/codes')
import xgb_model
import pickle
import pandas as pd
import time

datasets = ['breast_cancer', 'diabetes', 'covtype', 'cod_rna', 'HIGGS', 'ijcnn1', 'Sensorless', 'webspam', 'MNIST', 'Fashion_MNIST', 'MNIST2_6']
n = len(datasets)
nclasses = [2, 2,  7, 2,  2, 2, 11, 2, 10, 10, 2]
sample_size = [100, 100, 5000, 5000,  5000, 5000, 5000, 5000, 5000, 5000, 1000]
deps = [6,5,8,4,8,8,6,8, 8, 8, 4]
trees = [4,20,80,80,300,60,30,100, 200, 200, 1000]
parameters = []
for i in range(n):
    pdict = {'max_dep':deps[i], 'ntrees':trees[i]}
    parameters.append(pdict)


test_accuracies = []
used_time = []
for i in range(n):
    start = time.time()
    model, accuracy, sample_df = xgb_model.xgb_model(datasets[i], parameters[i], nclasses[i], sample_size[i])
    end = time.time()
    used_time.append(end - start)
    test_accuracies.append(accuracy)
    os.chdir('/home/cai7/chosen_sample/xgb')
    sample_df.to_pickle('{}_xgb_samples.pkl'.format(datasets[i]))
    os.chdir('/home/cai7/models/xgb')
    pickle.dump(model, open('{}_xgb_model.pkl'.format(datasets[i]), 'wb'))    
    print('{} is done'.format(datasets[i]))
    
    
accu_df = pd.DataFrame()
accu_df['datasets'] = datasets
accu_df['test accuracy'] = test_accuracies
accu_df['used time'] = used_time
os.chdir('/home/cai7/test_accu')
accu_df.to_excel('xgb_accu.xlsx')
accu_df.to_csv('xgb_accu.txt')


