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
from datetime import datetime
from sklearn.preprocessing import  MinMaxScaler, scale
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

def normal_data(x):
    # 列归一化
    # 若数值差距过大，则取log在归一化
#    for i in range(x.shape[1]):
#        m = np.max(x[:,i])
#        n = np.min(x[:,i])
#        if m // (n + 1e-10) > 100:
#            x[:, i] = np.log(x[:, i])
#            x[:, i] = MinMaxScaler().fit_transform(x[:,i])
#        else:
#            x[:, i] = MinMaxScaler().fit_transform(x[:, i])
    for i in range(x.shape[1]):
        x = MinMaxScaler().fit_transform(x)
    return x
    
    
    
       
def trans_data(datalist):
    # get X, Y and normal
    X = []
    Y = []
    for i in range(0, 3):#####range(len(datalist)):
        data = pd.read_csv(datalist[i], header=None, low_memory=False)
#        data = data.sort_values(2, ascending=0)
        data = data.values
        data = data.astype('float32')
        data = normal_data(data)
        for j in range(len(data)):
            X.append(data[j, 3:])
            Y.append(1 if j < int(len(data) * 0.35) else 0)
    X = np.array(X)
    Y = np.array(Y)
#    print(X.shape)
#    print(Y.shape)
#    XY = np.hstack((X, Y.T))
#    print(XY.shape)
#    np.random.shuffle(XY)
#    X = XY[:,:-2]
#    Y = XY[:,-1]
    return X, Y
    
def infer_rank(model, t='C:\\Users\\Jhy\\Desktop\\data\\501.csv'):
    # inference and rank it, return metric 
    x = pd.read_csv(t, header=None)
    x = x.values()
    x = x.astype('float32')
    result = np.zeros([len(), 2])
    for i in range(len(x)):
        result[i, 0] = x[i, 0]
        result[i, 1] = model.predict_prob(x[i, 3:])
    result[result[:, 1].argsort()]
    ID_select = result[0:50, 1]
    re = []
    for i in range(len(x)):
        for ID in ID_select:    
            if x.iloc[i, 0] == ID:
                re.append(i)
    return sum(re) / len(re)
           
        
        
    
def ID_to_label(ID=None,
                root='C:\\Users\\Jhy\\Desktop\\data\\1.csv'):
    # 读取  某t某个ID的 的收益/ 排名
#    root = 'C:\\Users\\Jhy1993\\Desktop\\data\\1.csv'
#    filename = locals()[str(ID) + '.csv'] 
#    filename =    
#    filepath = os.path.join(root, filename)
    x = pd.read_csv(root, header=None)
    #x2 = x.loc[i] if x
#    x1 = x.sort_values(0)
#    xx = x.iloc[1,0]
    for i in range(len(x)):
        if x.iloc[i, 0] == ID:
            x2 = x.iloc[i, 1]
    return x
#jhy = ID_to_label(501)
#    
#for t in range(1, 2):
#    filename = locals()[str(t) + '.csv']
#    filepath = os.path.join(root, filename)
#    data = pd.read_csv(filepath, header=None)

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
    model.add(Dropout(0.5))
    
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    sgd = SGD(lr=0.0005, decay=1e-6, momentum=0.9, nesterov=True)
    #model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, Y_train, X_test, Y_test, batch_size=128,
                nb_epoch=100):
    model.fit(X_train, Y_train, batch_size=batch_size, shuffle=True,
              nb_epoch=nb_epoch, verbose=2, validation_data=[X_test, Y_test])
    score, acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test Score: {}  Test Accuracy: {}'.format(score, acc))
    return model

def save_model(model,
               ModelPath='model_' + datetime.now().strftime("%m%d_%H%M") + '.json',
               WeightPath='model_' + datetime.now().strftime("%m%d_%H%M") + '.h5'):
    print('start save model...')
    json_string = model.to_json()
    fd = open(ModelPath, 'w')
    fd.write(json_string)
    fd.close()
    model.save_weights(WeightPath)
    print('model is saved.')

#def pred(model,    ....):
#    x_path = os.path.join()
#    x = get_data(root)
#    pass

 
if __name__ == '__main__':
    t1 = time.time()
#    root = 'C:\\Users\\Jhy\\Desktop\\data'
    root = '../data'
    ModelPath = 'model_' + datetime.now().strftime("%m%d_%H%M") + '.json'
    WeightPath = 'model_' + datetime.now().strftime("%m%d_%H%M") + '.h5'

    data_train, data_test = get_data(root)
    X_train, Y_train = trans_data(data_train)           
    X_test, Y_test = trans_data(data_test)
    '''
    model = DNN()
    train_model(model, X_train, Y_train, X_test, Y_test,
                batch_size=256, nb_epoch=100)
    save_model(model)
    t2 = time.time()
    print('Time Cost: {}'.format(t2 - t1))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# split data
def split_data(X, Y, rate=0.7):
    sp = int(len(X) * rate)
    X_train = X[0:sp]
    Y_train = Y[0:sp]
    X_test = X[sp:]
    Y_test = Y[sp:]
    return X_train, Y_train, X_test, Y_test
'''
    
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
    
    
    
    
    
    
    
    