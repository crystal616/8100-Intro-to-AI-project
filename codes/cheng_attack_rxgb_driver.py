# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:14:27 2019

@author: Ying
"""
import pandas as pd
import xgboost as xgb
import numpy as np
import os
os.chdir('/home/cai7/codes')
import cheng_attack_rxgb
from sklearn.datasets import load_svmlight_file

dataset = 'breast_cancer'
nclasses = 2
n_features = 9
binary = False
if nclasses == 2:
    binary = True

bst = xgb.Booster()
model_path = '/home/cai7/models/rxgb/{}/{}_rxgb.model'.format(dataset, dataset)
bst.load_model(model_path)
test_data, test_label = load_svmlight_file('/home/cai7/chosen_sample/rxgb/{}_rxgb_samples.s'.format(dataset), n_features = n_features)
test_data = test_data.toarray()
test_label = test_label.astype('int')
if len(test_label) >= 1000:
    test_data = test_data[:200]
    test_label = test_label[:200]

ori_points = []
results = []
for i in range(len(test_label)):
    s = test_data[i]
    sl = test_label[i]
    r = cheng_attack_rxgb.attack(bst, test_data, test_label, s, sl, nclasses, i)
    ori_points.append(s)
    results.append(r)
    print('{} is done'.format(i))

total_dis = 0
pert = pd.DataFrame()
index = []
points = []
dis = []
for (i, d, p) in results:
    index.append(i)
    points.append(p)
    dis.append(d)
    total_dis += d

pert['index'] = index
pert['distance'] = dis
pert['pert point'] = points
pert['ori point'] = ori_points
os.chdir('/home/cai7/attack/cheng')
pert.to_csv('{}_cheng_attack_rxgb.txt'.format(dataset))
with open('{}_cheng_rxgb_ave.txt'.format(dataset), 'w') as f:
    f.write('average distance: ' + str(total_dis/len(test_label)))


f.close()