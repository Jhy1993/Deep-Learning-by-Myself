# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:38:28 2016

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
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

def ID_to_label(ID,
                root='C:\\Users\\Jhy\\Desktop\\data\\1.csv'):
    # 读取  某t某个ID的 的收益/ 排名
#    root = 'C:\\Users\\Jhy1993\\Desktop\\data\\1.csv'
#    filename = locals()[str(ID) + '.csv'] 
#    filename =    
#    filepath = os.path.join(root, filename)
    x = pd.read_csv(root, header=None)
    #x2 = x.loc[i] if x
#    x1 = x.sort_values(0)
#    xx = x.iloc[1,0]
    for i in range(len(x)):
        if x.iloc[i, 0] == '222180117136533310.00000000':
#'ID:
            x2 = x.iloc[i, 2]
    return x2
jhy = ID_to_label(501)
 