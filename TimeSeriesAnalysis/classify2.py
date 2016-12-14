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
# =====采用data。txt的数据，存在问题：不同t的排名无法比较 所以抛弃=======
#def get_data(filepath = 'C:\\Users\\Jhy\\Desktop\\data\\1.csv'):    
#    data = pd.read_csv(filepath, header=None)
##    data2 = np.array(list(data))
#    data2 = np.array(data)
#    X = []
#    Y = []
#    print ('www')
#    for i in range(len(data2)):
#        if data2[i, 2] > 0.5:
#            X.append(data2[i, 3:])
#            Y.append(1)
#        else:
#            X.append(data2[i, 3:])
#            Y.append(0)   
#    X = np.array(X)
#    Y = np.array(Y)
#    
#    X_train, Y_train, X_test, Y_test = split_data(X, Y)
#    return X_train, Y_train, X_test, Y_test
# 
# 打标签需要对每个文件的排名分别打标签      



def get_data(root, split=500):
    data_train = []
    data_test = []
    for i in os.listdir(root):
        if os.path.isfile(os.path.join(root, i)):
    #        print (i)
    #        print (os.path.join(root, i))
            if int(i.split('.')[0]) <= split:
                data_train.append(os.path.join(root, i))
            else:
                data_test.append(os.path.join(root, i))
    return data_train, data_test

           
def trans_data(data):
    X = []
    Y = []
    for i in range(0, 2):#####range(len(data)):
        data = pd.read_csv(data_train[i], header=None)
        data = data.values
        data = data.astype('float32')
        for j in range(len(data)):
            X.append(data[j, 3:])
            Y.append(0 if data[j, 2] < 0.5 else 1)
    return X, Y

    
#for t in range(1, 2):
#    filename = locals()[str(t) + '.csv']
#    filepath = os.path.join(root, filename)
#    data = pd.read_csv(filepath, header=None)
#    
#dddd = pd.read_csv(d[0], header=None)
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
    #model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, Y_train, X_test, Y_test, batch_size=32,
                nb_epoch=100):
    model.fit(X_train, Y_train, batch_size=batch_size, shuffle=True,
              nb_epoch=nb_epoch)
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test, Score: {}  Test Accuracy: {}'.format(score, acc))
    return model

def save_model(model, )

    
if __name__ == '__main__':
    t1 = time.time()
    root = 'C:\\Users\\Jhy\\Desktop\\data'
    data_train, data_test = get_data(root)
    X_train, Y_train = trans_data(data_train)           
    X_test, Y_test = trans_data(data_test)
    Y_t
    model = DNN()
    train_model(model, X_train, Y_train, X_test, Y_test)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
    


# 对单个时间片进行排序
t = 测试机中某一个



def sort_metrics(t, model, folderpath= ):
    filename = locals()[str(t) + '.csv']
    filepath = os.path.join(folderpath, filename)
    data = pd.read_csv(filepath, header=None)
    isloc


    
    



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
    
    
    
    
    
    
    
    