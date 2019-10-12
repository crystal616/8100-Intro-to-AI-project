# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 16:27:01 2019

@author: ycai
"""

import os
os.chdir('/home/cai7/codes')
import dt_model
import pickle
import pandas as pd
import time

datasets = ['breast_cancer', 'diabetes', 'ionosphere']
n = len(datasets)
sample_size = [100, 100, 100]
deps = [5,5,4]

test_accuracies = []
used_time = []
for i in range(n):
    start = time.time()
    model, accuracy, sample_df = dt_model.dt_model(datasets[i], deps[i], sample_size[i])
    end = time.time()
    used_time.append(end - start)
    test_accuracies.append(accuracy)
    os.chdir('/home/cai7/chosen_sample/dt')
    sample_df.to_pickle('{}_dt_samples.pkl'.format(datasets[i]))
    os.chdir('/home/cai7/models/dt')
    pickle.dump(model, open('{}_dt_model.pkl'.format(datasets[i]), 'wb'))
    print('{} is done'.format(datasets[i]))
    
accu_df = pd.DataFrame()
accu_df['datasets'] = datasets
accu_df['test accuracy'] = test_accuracies
accu_df['used time'] = used_time
os.chdir('/home/cai7/test_accu')
accu_df.to_excel('dt_accu.xlsx')
accu_df.to_csv('dt_accu.txt')