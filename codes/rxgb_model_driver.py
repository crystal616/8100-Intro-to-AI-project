# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 13:07:24 2019

@author: Ying
"""

import os
os.chdir('/home/cai7/codes')
import rxgb_prep
import time
import subprocess
import pandas as pd
import random


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
    parameters.append(pdict)

test_accuracies = []
used_time = []
for i in range(n):
    rxgb_prep.make_conf(datasets[i], parameters[i])
    if nclasses[i] == 2:
        binary_class = True
    else:
        binary_class = False
    rxgb_prep.df2svm(datasets[i], binary_class)
    
    os.chdir('/home/cai7/data/{}'.format(datasets[i]))
    test_df = pd.read_pickle('{}_test_df.pkl')
    
    os.chdir('/home/cai7/models/rxgb')
    try:
        os.mkdir('{}'.format(datasets[i]))
    except OSError:
        pass
    
    os.chdir('/home/cai7/models/rxgb/{}'.format(datasets[i]))
    subprocess.call(['cd', '/home/cai7/models/rxgb/{}'.format(datasets[i])])
    start = time.time()
    subprocess.call(['/home/cai7/RobustTrees/xgboost', '/home/cai7/data/rxgb/{}.conf'.format(datasets[i])])
    output = subprocess.check_output(['ls'])
    t = str(output)
    t1 = t[t.index("'")+1:t.index("\\")]
    os.rename(t1, '{}_rxgb.model'.format(datasets[i]))
    model = 
    
    model, accuracy, sample_df = rxgb_model.xgb_model(datasets[i], parameters[i], nclasses[i], sample_size[i])
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