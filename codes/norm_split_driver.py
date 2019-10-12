# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:26:18 2019

@author: ycai
"""

import os
os.chdir('/home/cai7/codes')
import norm_split


datasets = ['breast_cancer', 'diabetes', 'ionosphere', 'covtype', 'cod_rna', 'Fashion_MNIST', 'HIGGS', 'ijcnn1', 'MNIST', 'Sensorless', 'webspam', 'MNIST_2_6']
n = len(datasets)
labels = [10, ]

file_name = ['breast-cancer-wisconsin.data', ]
full_size = [546+137, 614+154, 281+70, 400000+181000, 59535+271617, 70000, 10500000+500000, 49990+91701, 70000, 48509+10000, 350000, 11876+1990]
test_size = [137, 154, 70, 181000, 271617, 10000, 500000, 91701, 10000, 10000, 50000, 1990]
nfeatures = [9, ]
column_format = [0, ]
seporators = [',', ]
headers = [None]
missing = ['?', ]
drops = [[0], ]


for i in range(n):
    norm_split.norm_split(datasets[i], labels[i], file_name[i], test_size[i], full_size[i], nfeatures[i], column_format[i], seporators[i], headers[i], missing[i], drops[i])