# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:06:02 2016

@author: Jhy_BISTU
INPUT:

OUTPUT:

REFERENCE:
https://gist.github.com/hnykda/c362f0ad488e3b289394
"""
import pandas as pd
from random import random
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import LSTM, Dense, Activation

flow = (list(range(1, 10, 1)) + list(range(10, 1, -1))) * 100
pdata = pd.DataFrame({'a': flow, 'b': flow})
pdata.b = pdata.b.shift(9)
# Add noise
data = pdata.iloc[10:] * random() 
plt.plot(data[1:200])
plt.xlabel('t')
plt.ylabel('price')
plt.show()

def _load_data(data, n_prev = 100):
    '''
    data should be pd.Dataframe()
    '''
    
    
    return data

def train_test_split(df, test_size=0.1):
    '''
    split data based on test_size
    '''
    n_train = int(round(len(df) * (1 - test_size)))
    return n_train
    

    

















