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
import cheng_attack
import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())

dataset = 'breast_cancer'
model = xgb.Booster()
model_path = '/home/cai7/models/xgb/{}_xgb.model'.format(dataset)
model.load_model(model_path)
test_df = pd.read_pickle('/home/cai7/chosen_sample/xgb/{}_xgb_samples.pkl'.format(dataset))

test_data = np.array(test_df.drop(columns = ['label']))
test_label = test_df['label'].tolist()
dtest = xgb.DMatrix(test_data, label = test_label)

i = 2
s = test_data[i]
sl = test_label[i]
r = attack(model, test_data, test_label, s, sl, 2, i)


model, tdata, tlabel, x0, y0, nclasses, index = model, test_data, test_label, s, sl, 2, i
step = 0.2
beta = 0.001
iterations = 1000

q = 20    
nf = len(x0)    
best_theta, g_theta, dis = None, float('inf'), float('inf')

for i in range(len(tlabel)):
        if predict(model, tdata[i], nclasses) != y0:
            print(str(i))
            theta = tdata[i] - x0
            initial_lbd = 1.0            
            lbd, distance = fine_grained_binary_search(model, x0, y0, theta, initial_lbd, nclasses)
            if distance < dis:
                best_theta, g_theta, dis = theta, lbd, distance

theta = best_theta
pre_v = g_theta
stopping = 0.0005    
min_dis = dis
min_theta = theta
min_v = pre_v
count = 0







(model, tdata, tlabel, x0, y0, nclasses, index, step = 0.2, beta = 0.001, iterations = 1000)

results = [pool.apply_async(cheng_attack.attack, args=(model, test_data[i], test_label[i], 0, 2)) for i in range(len(test_label))]
    
pool.close()
pool.join()