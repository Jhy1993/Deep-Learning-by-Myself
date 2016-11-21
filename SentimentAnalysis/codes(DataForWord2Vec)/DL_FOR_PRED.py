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
class DL_for_predict:
    """docstring for DL_for_predict"""
    def __init__(self,
                MAX_SEQUENCE_LENGTH = 100,
                word2vec_path = 'wiki.zh.text.vector',
                ModelPath = 'model_3c_0824.json',
                WeightPath = 'model_3c_0824.h5',
                word_index_path = 'word_index.pkl'
                ):
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH#100
        self.word2vec_path = word2vec_path#'wiki.zh.text.vector'
        self.ModelPath = ModelPath#'model_3c_0822.json'
        self.WeightPath = WeightPath#'model_3c_0822.h5'
        self.word_index_path = word_index_path
    def load_word_index(self):
        print ('start load word_index.........')
        wi = open(self.word_index_path, 'rb')
        word_index = pickle.load(wi)
        wi.close()
        print ('load word_index is ok!!!')
        return word_index
    def load_LSTM_model(self):
        model = model_from_json(open(self.ModelPath).read())  
        model.load_weights(self.WeightPath) 
        print ('deep learning model is ok!!!')
        return model 
    def doc2num(self, s, word_index): 
        s = [i for i in s if i in word_index.index]
        s = s[:self.MAX_SEQUENCE_LENGTH] + ['']*max(0, self.MAX_SEQUENCE_LENGTH-len(s))
        return list(word_index[s])
    def preprocess(self, s, word_index):
        s = np.array(self.doc2num(list(jieba.cut(s)), word_index))
        s = s.reshape((1, s.shape[0]))
        return s
    def GO(self, s):
        word_index = self.load_word_index()
        model = self.load_LSTM_model() 
        sp = self.preprocess(s, word_index)
        predict_label = model.predict_classes(sp)
        return predict_label
if __name__ == '__main__':  
    s = '款式不错，就是有点大'
    DLP = DL_for_predict()
    predict_label = DLP.GO(s)
    print s, '---label is ---', predict_label