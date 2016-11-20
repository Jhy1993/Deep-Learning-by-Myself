# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:02:43 2016

@author: box1
"""
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import os
from PIL import Image
import time
import pandas as pd
import jieba
import numpy as np
import pickle
import json
import h5py
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop 
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm 
from keras.regularizers import l2, activity_l2
class DL:
    """docstring for DL"""
    def __init__(self, 
        filedir= 'C:\\Users\\box1\\Desktop\\picture_DL\\2'
        ):
        self.filedir = filedir

    def get_data(self):
        #include data and label
        ALL_data = []
        ALL_label = []
        filelist = os.listdir(self.filedir)
        for file in filelist:
            im = Image.open(os.path.join(self.filedir, file)).convert('L')
            data = im.load()
            #data = np.matrix(data)

            label = file.split('_')
            label = label[0].lower()
            ALL_label.append(label)
            ALL_data.append(data)
#            print ALL_data
#            print label
        #data.show()
        return ALL_data, ALL_label
    def transform_format(data, label):
        pass
    def CNN_model(data, label, split=0.3):
        print ('start CNN model...')
        model = Sequential()

        model.add(Convolution2D(64, 5, 5, 
            border_mode='valid', input_shape=data.shape[-3:]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(32, 5, 5, border_mode='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512, init='normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(nb_labels, init='normal'))
        model.add(Activation('softmax'))
        sgd = SGD(l2=0.0, lr=0.01, decay=1e-6,
            momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy',
            optimizer=sgd, class_mode="categorical")
        early_Stop = EarlyStopping(monitor='acc',#'val_loss', 
                                    verbose=2, mode='auto', patience=10)
        batch_size = 32
        train_num = int(len(x) * 0.8)
        model.fit(x[:train_num], y[:train_num], 
               batch_size = batch_size, 
                shuffle=True, 
                verbose = 2,
                #validation_split=0.1,
                validation_data=[x[train_num:], y[train_num:]],
                #callbacks=[early_Stop],
              nb_epoch=self.nb)
        print ('CNN model training is ok!!!')
        #测试模型
        score, acc = model.evaluate(x[train_num:], y[train_num:], 
                                batch_size = batch_size)
        print('Test accuracy:', acc)
        return model       
if __name__ == '__main__':
    DL = DL()
    data, label = DL.get_data()
    print data[1]
        
