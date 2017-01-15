# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:

INPUT:

OUTPUT:

REFERENCE:

"""
from __future__ import print_function

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization



root = 'C:\\Users\\Jhy\\Desktop\\data\\1.csv'
data = pd.read_csv(root, header=None)
data_descirbe = data.describe()
fea_des = data_descirbe.iloc[:, range(3, data_descirbe.shape[1])]

#=======================离群点检测， 针对单个变量============================================
n = 295 # 实际是特征1 
fea = data.iloc[:, n]
plt.figure()
plt.plot(fea)
plt.title('col 295 ')
plt.figure()
fea.plot(kind='kde')
plt.title('col 295 Probability Density')

# 处理一下
fea_des = fea.describe()
n_sigma = 2
upbound = fea_des.ix[1] + n_sigma * fea_des[2]
lowbound = fea_des.ix[1] - n_sigma * fea_des[2]
fea[fea > upbound ] = 136
fea[fea < lowbound] = 1

plt.figure()
plt.plot(fea)
plt.title('col 295 -processed')
plt.figure()
fea.plot(kind='kde')
plt.title('col 295 Probability Density -processed')


#===================================std======================
#x_std = fea_des.ix[2]
#x_std = np.log(x_std)
#x_std2 = x_std[x_std < 5]
#plt.figure()
#plt.plot(x_std, 'k--')
#plt.title('std')

#plt.figure()
#plt.plot(x_std2, 'k--')
#xlim(0, 800)
#plt.title('std < 5')
#
#
#xmax = ss.ix[7]
#xmin = ss.ix[3]
#plt.figure()
#plt.plot(xmax, 'k--')
#plt.plot(xmin, 'g--')


# ===========================皮尔森系数计算相关性================================
'''
np.random.seed(0)
size = 1000
x = np.random.normal(0, 1, size)
print stats.pearsonr(x, x + np.random.normal(0, 1, size))
print stats.pearsonr(x, x + np.random.normal(0, 100, size))

'''
#===========================对std筛选， 特别大的， 特别小的=======================================
'''
x_std = x_std.values
x_std = x_std.astype('float32')
print('方差小于0.001的列序号')
std1 = np.where(x_std < 0.01)
std1 = np.asarray(std1)
print('共有 {} 个, 分别为：{}'.format(std1.shape, std1))
'''
#=========================画出max/min 的图，筛选出过大的列=========================================
'''
bs = xmax / (xmin + 1e-10)
plt.figure()
#xlim(250, 350)
#ylim(0, 7e15)
plt.plot(bs, 'k--')
plt.title('max  / min ')
#xmin = xmin.values
#xmin = xmin.astype('float32')
bs = bs.values
bs = bs.astype('float32')

col = np.where(bs > 100)
col = np.asarray(col)
print('max / min > 100 的列有{}， 分别是{}'.format(col.shape, col))
'''











