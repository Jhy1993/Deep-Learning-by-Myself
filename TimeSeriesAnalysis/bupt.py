# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:28:24 2016

@author: Jhy_BISTU
INPUT:

OUTPUT:

REFERENCE:

"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from keras.models import Sequential
#from keras.layers import Dense, LSTM, Dropout, Activation
# Data
#data = pd.read_csv('C:\\Users\\Jhy\\Desktop\\Deep Learning Code\\TimeSeriesAnalysis\\tsp.csv')
dataframe = pd.read_csv('C:\\Users\\Jhy\\Desktop\\Deep Learning Code\\TimeSeriesAnalysis\\tsp.csv')

ID = data[:, 1]
Time = data[:, 2]
Rank = da
Features = []


data_dim = ?
timesteps = ?
nb_classes = ?
# Model
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=64, nb_epoch=5,
          validation_data=(x_val, y_val))
