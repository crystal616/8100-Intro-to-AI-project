# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 17:13:31 2019

@author: Ying
"""

import os
os.chdir('/home/cai7/codes')
import rxgb_prep
import time
import subprocess
import pandas as pd
import random

datasets = ['breast_cancer', 'diabetes', 'covtype', 'cod_rna', 'HIGGS', 'ijcnn1', 'Sensorless', 'webspam', 'MNIST', 'Fashion_MNIST', 'MNIST2_6']
n = len(datasets)
nclasses = [2, 2,  7, 2,  2, 2, 11, 2, 10, 10, 2]
sample_size = [100, 100, 5000, 5000,  5000, 5000, 5000, 5000, 5000, 5000, 1000]
deps = [8,5,8,5,8,8,6,8, 8, 8, 6]
trees = [20,20,80,80,300,60,30,100, 200, 200, 1000]
eps = [0.3, 0.2, 0.2, 0.2, 0.05, 0.1, 0.05, 0.05, 0.3, 0.1, 0.3]
parameters = []
for i in range(n):
    pdict = {'max_dep':deps[i], 'ntrees':trees[i], 'eps':eps[i]}
    if nclasses[i] == 2:
        pdict['objective'] = 'binary:logistic'
    else:
        pdict['objective'] = 'multi:softmax'
        
    if nclasses[i] > 2:
        pdict['num_class'] = nclasses[i]
    parameters.append(pdict)

times = []
for i in range(n):
    rxgb_prep.make_conf(datasets[i], parameters[i])
    if nclasses[i] == 2:
        binary_class = True
    else:
        binary_class = False
    rxgb_prep.df2svm(datasets[i], binary_class)
    
    os.chdir('/home/cai7/models/rxgb')
    try:
        os.mkdir('{}'.format(datasets[i]))
    except OSError:
        pass
    
    os.chdir('/home/cai7/models/rxgb/{}'.format(datasets[i]))
    subprocess.call(['cd', '/home/cai7/models/rxgb/{}'.format(datasets[i])])
    start = time.time()
    subprocess.call(['/home/cai7/RobustTrees/xgboost', '/home/cai7/data/rxgb/{}.conf'.format(datasets[i])])
    end = time.time()
    times.append(end - start)
    output = subprocess.check_output(['ls'])
    t = str(output)
    t1 = t[t.index("'")+1:t.index("\\")]
    os.rename(t1, '{}_rxgb.model'.format(datasets[i]))
    print('{} is done'.format(datasets[i]))

t_df = pd.DataFrame()
t_df['dataset'] = datasets
t_df['used time'] = times
os.chdir('/home/cai7/test_accu')
t_df.to_csv('rxgb_used_time.txt')
t_df.to_excel('rxgb_used_time.xlsx')