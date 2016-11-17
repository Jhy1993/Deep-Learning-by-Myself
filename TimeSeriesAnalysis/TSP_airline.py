# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:06:02 2016

@author: Jhy_BISTU
INPUT:

OUTPUT:

REFERENCE:
http://machinelearningmastery.com/time-series-prediction-with-deep-learning-in-python-with-keras/
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(7)

dataset = pd.read_csv('C:\\Users\\Jhy\\Desktop\\Deep Learning Code\\TimeSeriesAnalysis\\international-airline-passengers.csv',
                      usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()
dataset = dataset.values
dataset = dataset.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.7)
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]


def create_dataset(dataset, look_back=1):
    X = []
    Y = []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:i+look_back, 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input from [samples, features] to [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
