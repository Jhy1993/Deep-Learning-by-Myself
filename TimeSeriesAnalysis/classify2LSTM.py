# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 19:31:10 2016

README:
Classify price into 2 classes use  LSTM
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


def get_data(root):
    full = []
    for i in os.listdir(root):
        filepath = os.path.join(root, i)
        if os.path.isfile(filepath):
            data = pd.read_csv(filepath, header=None, low_memory=False)
            full.append(data)
    full = pd.concat(full, axis=0)
    return full

def lstm_data(x):
    # x为所有csv的拼接
    ID_all = set(x.)
    
            


if __name__ == '__main__':
    root = 'C:\\Users\\Jhy\\Desktop\\d2'
    data = get_data_LSTM(root)