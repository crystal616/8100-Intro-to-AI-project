# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 18:14:27 2019

@author: Ying
"""
import pandas as pd
import xgboost as xgb
import numpy as np
import os
#os.chdir('../codes')
import cheng_attack
from sklearn.model_selection import train_test_split

dataset = 'covtype'
nclasses = 7
model = xgb.Booster()
model_path = '../models/xgb/{}_xgb.model'.format(dataset)
model.load_model(model_path)
test_df = pd.read_pickle('../chosen_sample/xgb/{}_xgb_samples.pkl'.format(dataset))
if test_df.shape[0] >= 1000:
    _, test_df = train_test_split(test_df, test_size = 200)


test_df = test_df.reset_index(drop=True)
test_data = np.array(test_df.drop(columns = ['label']))
test_label = test_df['label'].tolist()
dtest = xgb.DMatrix(test_data, label = test_label)

ori_points = []
results = []
for i in range(len(test_label)):
    s = test_data[i]
    sl = test_label[i]
    r = cheng_attack.attack(model, test_data, test_label, s, sl, nclasses, i)
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
os.chdir('../attack/cheng')
pert.to_csv('{}_cheng_attack_xgb.txt'.format(dataset))
with open('{}_cheng_xgb_ave.txt'.format(dataset), 'w') as f:
    f.write('average distance: ' + str(total_dis/len(test_label)))
print("average_distance: ",total_dis/len(test_label))

  

f.close()