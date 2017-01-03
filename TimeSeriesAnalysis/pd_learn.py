# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:

INPUT:

OUTPUT:

REFERENCE:

"""
#import tensorflow as tf
import numpy as np
import pandas as pd
import os

root = 'C:\\Users\\Jhy1993\\Desktop\\data'
data = pd.DataFrame()
filelist = []
for i in os.listdir(root):
    print(i)
    filepath = os.path.join(root, i)
    temp = pd.read_csv(filepath, header=None, low_memory=False)
    filelist.append(temp)
data = pd.concat(filelist, ignore_index=True)
ID_count = data[0].value_counts()
ID = ID_count.index
ID = np.asarray(ID)
print(ID[3])
dt = data.ix[data[0] == ID[0]] 
#print(dt)

'''
将某个id数据全部筛选出来 ，data.ix[data[0] == 'id']
然后对t设为index， obj.reindex(np.range(600), fill_value=0.5)
这里假设 fill-index将空特征设为0.5
或者，method=‘ffill’——

root = 'C:\\Users\\Jhy1993\\Desktop\\1.csv'
data = pd.read_csv(root, header=None)
data2 = data[:, 1:]
data2.reindex(np.arange(10), fill_value=0.5)
'''













