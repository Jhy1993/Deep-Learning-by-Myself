# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:51:23 2016

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


num = 10
re = np.zeros([num, 2])
for i in range(num):
    re[i, 0] = i
    re[i, 1] = i*2
    
a = np.array([[1,4],[0.3,5], [0, 0]])
print(a)
aa = np.sort(a, axis=0)
print(a[1:2, 1])
#print(aa)

aaa = a[a[:,0].argsort()]
#print(aaa)
