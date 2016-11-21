# -*- coding: utf-8 -*-
"""
Created on  Aug 22 15:03:08 2016

@author: Jhy_Bistu
"""
# from DL_FOR_TRAIN import DL_for_train
from DL_FOR_TRAIN_V8 import DL_for_train
import time
t1 = time.time()

data_path = 'comments_2016-08-08_v4.xls'
MAX_SEQUENCE_LENGTH = 100 
word2vec_path = 'wiki.zh.text.vector'
nb = 50
min_count = 5  
ModelPath = 'model_3c_0819.json'
WeightPath = 'model_3c_0819.h5'
word_index_path = 'word_index.pkl'
DL = DL_for_train(data_path=data_path,
            MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
            word2vec_path=word2vec_path,
            nb=nb,
            min_count = min_count,
            ModelPath=ModelPath,
            WeightPath=WeightPath,
            word_index_path=word_index_path)
# DL = DL_model()
model = DL.GO()

t2 = time.time()
print ('本次运行时间：%.1d 秒' %(t2-t1) )