#!/usr/bin/env python
# -*- coding: utf-8 -*-
#---------------load model------------ 
import numpy as np
import pandas as pd
import jieba
import re
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop 
import json
import h5py
from keras.models import model_from_json 
import pickle
def load_word_index(wiPath):
    print ('start load word_index.........')
    wi = open(wiPath, 'rb')
    word_index = pickle.load(wi)
    wi.close()
    print ('load word_index is ok!!!')
    return word_index
def load_LSTM_model(modelPath, weightPath):
    pass
    print ('start load deep learning model...')
    model = model_from_json(open(modelPath).read())  
    model.load_weights(weightPath) 
    print ('deep learning model is ok!!!')
    return model 

wiPath = 'word_index.pkl'
modelPath = 'model_3c_0813.json'
weightPath = 'model_3c_0813.h5'


#---------------------------predict------------

MAX_SEQUENCE_LENGTH = 100
def doc2num(s, MAX_SEQUENCE_LENGTH, word_index): 
    s = [i for i in s if i in word_index.index]
    s = s[:MAX_SEQUENCE_LENGTH] + ['']*max(0, MAX_SEQUENCE_LENGTH-len(s))
    return list(word_index[s])
def predict_one(s): #单个句子的预测函数
    word_index = load_word_index(wiPath)
    model = load_LSTM_model(modelPath, weightPath)
    s = np.array(doc2num(list(jieba.cut(s)), MAX_SEQUENCE_LENGTH, word_index))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s)

s1 = '款式不错，就是有点大'
predict_label1 = predict_one(s1)
print s1, '---label is ---', predict_label1