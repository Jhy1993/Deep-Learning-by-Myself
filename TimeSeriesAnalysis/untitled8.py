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
x = pd.read_csv(root, header=None)
#xx = x.sort()
#x2 = x.loc[i] if x
x1 = x.sort_values(2, ascending=0)
i = int(x1.shape[0] * 0.35)
x_zheng = x1[0:i]
xx = x.iloc[1,2]
#for i in range(len(x)):
#    if x.iloc[i, 0] == 5:
#        x2 = x.iloc[i, 1]
print(x2)
#


















