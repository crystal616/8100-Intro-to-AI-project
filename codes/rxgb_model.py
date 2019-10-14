# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 10:36:05 2019

@author: Ying
"""


        
import os
os.chdir('/home/cai7/codes')
import rxgb_prep
import subprocess


dataset = 'diabetes'
parameters = {'objective':'binary:logistic', 'eps':0.2, 'max_dep': 5, 'ntrees':20}
binary_class = True

dataset = 'breast_cancer'
parameters = {'objective':'binary:logistic', 'eps':0.3, 'max_dep': 8, 'ntrees':4}
binary_class = True

dataset = 'Sensorless'
parameters = {'objective':'multi:softmax', 'eps':0.05, 'max_dep': 6, 'ntrees':30}
binary_class = False

rxgb_prep.make_conf(dataset, parameters)

rxgb_prep.df2svm(dataset, binary_class)

os.chdir('/home/cai7/models/rxgb')
try:
    os.mkdir('{}'.format(dataset))
except OSError:
    pass

os.chdir('/home/cai7/models/rxgb/{}'.format(dataset))

subprocess.call(['cd', '/home/cai7/models/rxgb/{}'.format(dataset)])

subprocess.call(['/home/cai7/RobustTrees/xgboost', '/home/cai7/data/rxgb/{}.conf'.format(dataset)])
output = subprocess.check_output(['ls'])
t = str(output)
t1 = t[t.index("'")+1:t.index("\\")]
os.rename(t1, '{}_rxgb.model'.format(dataset))


    