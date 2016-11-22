# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:19:36 2016

@author: Jhy_BISTU
README：
项目分析： 分类预测 features
INPUT:

OUTPUT:

REFERENCE:

"""

from keras.models import Sequential
from keras.layers import LSTM, Dense

import numpy as np

data_dim = 16
timesteps = 8
nb_classes = 10

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  
model.add(LSTM(32, return_sequences=True))  
model.add(LSTM(32))  
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=64, nb_epoch=5,
          validation_data=(x_val, y_val))

# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: ', trainScore)
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: ', testScore)

# generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)