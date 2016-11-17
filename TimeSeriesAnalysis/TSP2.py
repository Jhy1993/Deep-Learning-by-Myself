# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 21:19:36 2016

@author: Jhy_BISTU
INPUT:

OUTPUT:

REFERENCE:

"""

import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation


class NeuralNetwork():
    def __init__(self, **kwargs):
        self.output_dim = kwargs.get('output_dim', 8)
        self.dropout = kwargs.get('dropout', 0.2)
        self.optimizer = kwargs.get('optimizer', 'rmsprop')
        self.num_lstm = kwargs.get('num_lstm', 3)
        self.loss = kwargs.get('loss', 'mse')
        self.batch_size = kwargs.get('batch_size', 32)

    def NN_model(self, x_train, y_train, x_test, y_test):
        print('Training on LSTM model...')
        input_dim = x_train.shape[1]
        model = Sequential()
        model.add(LSTM(ouput_dim=self.output_dim,
                       input_dim=input_dim,
                       dropout_U=self.dropout,
                       return_sequences=True))
        for i in range(self.lstm_layer - 1):
            model.add(LSTM(output_dim=self.output_dim,
                           input_dim=self.output_dim,
                           dropout_U=self.dropout,
                           return_sequence=True))
        model.add(Dense(output_dim=1,
                        input_dim=self.output_dim,
                        return_sequences=True))
        model.compile(loss=self.loss,
                      optimizer=self.optimizer,
                      batch_size=self.batch_size,
                      validation_data=(x_test, y_test))
        return model
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        


