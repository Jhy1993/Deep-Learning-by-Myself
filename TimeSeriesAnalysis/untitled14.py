# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:28:06 2016

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

import numpy as np
###矩阵a
a=np.floor(10*np.random.rand(5,2))
print(a)
###a

###矩阵b
b=np.floor(10*np.random.rand(5,1))
print(b)
ab = np.hstack((a, b))
#print(ab)
np.random.shuffle(ab)
#print(ab)
a_n = ab[:,0:a.shape[1]]
print(a_n)
b_n = ab[:, a.shape[1]]
print(b_n)
