# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:38:06 2016

@author: Jhy_BUPT
README:

INPUT:

OUTPUT:

REFERENCE:

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplitlib.pyplot as plt

dir= 'C:\Users\Jhy\Desktop\dd\dp'
files = os.listdir(dir)
data = pd.DataFrame()
for file in files:
    temp = pd.read_csv(os.path.join(dir, file), header=None)
    temp[1] = int(file.split('.')[0])
    data = pd.concat([data, temp], axis=0)

ID_count = data[0].value_counts()

x = data[data[0] == ID_count.index[0]]
plt.figure










