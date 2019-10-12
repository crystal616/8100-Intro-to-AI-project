# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:38:21 2019

@author: ycai
"""

from mlxtend.data import loadlocal_mnist
import pandas as pd
import os
import numpy as np

def normalize_data(X):
    features = X.columns.tolist()
    new_df = pd.DataFrame(columns = features)
    for f in features:
        current = X[f].tolist()
        min_v = np.min(current)
        max_v = np.max(current)
        delta = max_v - min_v
        if delta != 0:
            temp = (current - min_v) / delta
        else:
            temp = current
        new_df[f] = temp
    return new_df

X, y = loadlocal_mnist(
        images_path='/home/cai7/data/MNIST_orig/train-images-idx3-ubyte', 
        labels_path='/home/cai7/data/MNIST_orig/train-labels-idx1-ubyte')

df = pd.DataFrame(X)
X = df.astype('float')
X = normalize_data(X)
df = X
df['label'] = y.astype('int')

os.chdir('/home/cai7/data')
try:
    os.mkdir('MNIST')
except OSError:
    pass

os.chdir('/home/cai7/data/MNIST')
df.to_pickle('MNIST_train_df.pkl')

df_2 = df[df['label'] == 2]
df_6 = df[df['label'] == 6]
df_26 = pd.concat([df_2, df_6])
df_26 = df_26.reset_index(drop = True)

os.mkdir('/home/cai7/data/MNIST2_6')
os.chdir('/home/cai7/data/MNIST2_6')
df_26.to_pickle('MNIST2_6_train_df.pkl')


X, y = loadlocal_mnist(
        images_path='/home/cai7/data/MNIST_orig/t10k-images-idx3-ubyte', 
        labels_path='/home/cai7/data/MNIST_orig/t10k-labels-idx1-ubyte')

df = pd.DataFrame(X)
X = df.astype('float')
X = normalize_data(X)
df = X
df['label'] = y.astype('int')
os.chdir('/home/cai7/data/MNIST')
df.to_pickle('MNIST_test_df.pkl')

df_2 = df[df['label'] == 2]
df_6 = df[df['label'] == 6]
df_26 = pd.concat([df_2, df_6])
df_26 = df_26.reset_index(drop = True)

os.chdir('/home/cai7/data/MNIST2_6')
df_26.to_pickle('MNIST2_6_test_df.pkl')

os.chdir('/home/cai7/data/Fashion_MNIST_orig')
df = pd.read_csv('fashion-mnist_train.csv')
y = df['label']
X = df.drop(columns = ['label'])
X = X.astype('float')
X = normalize_data(X)
df = X
df['label'] = y.astype('int')


os.chdir('/home/cai7/data')
try:
    os.mkdir('Fashion_MNIST')
except OSError:
    pass

os.chdir('/home/cai7/data/Fashion_MNIST')
df.to_pickle('Fashion_MNIST_train_df.pkl')

os.chdir('/home/cai7/data/Fashion_MNIST_orig')
df = pd.read_csv('fashion-mnist_test.csv')
y = df['label']
X = df.drop(columns = ['label'])
X = X.astype('float')
X = normalize_data(X)
df = X
df['label'] = y.astype('int')

os.chdir('/home/cai7/data/Fashion_MNIST')
df.to_pickle('Fashion_MNIST_test_df.pkl')