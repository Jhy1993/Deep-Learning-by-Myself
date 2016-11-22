# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:58:53 2016

@author: Jhy_BUPT
README:
Stacked LSTM for international airline passengers problem with memory
INPUT:

OUTPUT:

REFERENCE:

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back), 0]
        dataX,append(a)
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)

# Reshap intp X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input into [samples, timesteps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.reshape[1], 1))
# Fit LSTM
batch_size = 1
model = Sequential()
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1),
               stateful=True, return_sequences=True)
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1),
               stateful=True))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
for i in range(100):
    model.fit(trainX, tranY, nb_epoch=1, batch_size=batch_size,
              verbose=2, shuffle=False)
    model.reset_states()

trainPredict = model.predict(trainX, batch_size=batch_size)
model.reset_states()#??????????????
testPredict = model.predict(testX, batch_size=batch_size)

    
















                    