# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:05:41 2016

@author: Jhy_BUPT
README:
Classify price into 2 classes use  DNN
INPUT:

OUTPUT:

REFERENCE:

"""
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
import time

#====================1. getdata===============================================
# =====采用data。txt的数据，可能存在问题：不同t的排名无法比较 所以抛弃=======
def get_data(filepath = 'C:\\Users\\Jhy\\Desktop\\data\\1.csv'):    
    data = pd.read_csv(filepath, header=None)
#    data2 = np.array(list(data))
    data2 = np.array(data)
    X = []
    Y = []
    print ('www')
    for i in range(len(data2)):
        if data2[i, 2] > 0.5:
            X.append(data2[i, 3:])
            Y.append(1)
        else:
            X.append(data2[i, 3:])
            Y.append(0)   
    X_train, Y_train, X_test, Y_test = split_data(X, Y)
    return X_train, Y_train, X_test, Y_test
#X_train, Y_train, X_test, Y_test = get_data()    
# 打标签需要对每个文件的排名分别打标签      




# split data
def split_data(X, Y, rate=0.7):
    sp = int(len(X) * rate)
    X_train = X[0:sp]
    Y_train = Y[0:sp]
    X_test = X[sp:]
    Y_test = Y[sp:]
    return X_train, Y_train, X_test, Y_test
    
'''
#=========================2. 定义模型========================================
def DNN(input_dim=763):
    model = Sequential()
    
    model.add(Dense(2000, input_dim=input_dim))
    
    model.add(Dense(1000))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(epsilon=1e-04))
    
    model.add(Dense(500))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(epsilon=1e-04))
    
    model.add(Dense(200))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model
    



#==========================================================================
if __name__ == '__main__':
    t1 = time.time()
    model = DNN()
    X_train, Y_train, X_test, Y_test = get_data()
    model.fit(X_train, Y_train, batch_size=10, nb_epoch=1)
    model.predict_proba()
    predict_proba(self, x, batch_size=32, verbose=1)
    #这里以时间片t=1 为例， 测试
    t2 = time.time()
    print ('本次运行时间：{} 秒'.format(t2-t1))
'''
    
    
    
    
    
    
    
    