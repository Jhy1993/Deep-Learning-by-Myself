# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 14:15:27 2016
2分类： 1 好评
        0 差评
@author: Jhy
"""

import numpy as np
import pandas as pd
import jieba
 
pos = pd.read_excel('pos3.xls', header=None)
pos['label'] = 1
neg = pd.read_excel('neg3.xls', header=None)
neg['label'] = 0
all = pos.append(neg, ignore_index=True)
all['words'] = all[0].apply(lambda s: list(jieba.cut(s))) #调用结巴分词
 
maxlen = 100 #截断词数
min_count = 5 #出现次数少于该值的词扔掉。这是最简单的降维方法
 
content = []
for i in all['words']:
    content.extend(i)
 
abc = pd.Series(content).value_counts()
abc = abc[abc >= min_count]
abc[:] = range(1, len(abc)+1)
abc[''] = 0 #添加空字符串用来补全

def doc2num(s, maxlen): 
    s = [i for i in s if i in abc.index]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    return list(abc[s])
 
all['doc2num'] = all[0].apply(lambda s: doc2num(s, maxlen))
 
#手动打乱数据
idx = range(len(all))
np.random.shuffle(idx)
all = all.loc[idx]
 
#按keras的输入要求来生成数据
x = np.array(list(all['doc2num']))
y = np.array(list(all['label']))
y = y.reshape((-1,1)) #调整标签形状
 

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.utils import np_utils
from keras.optimizers import SGD
#建立模型
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen))

model.add(LSTM(128,return_sequences=True)) 
model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))


model.add(Dense(1))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

 
batch_size = 32
train_num = 600
 
model.fit(x[:train_num], y[:train_num], 
          batch_size = batch_size, nb_epoch=2)
 
score, acc = model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)
print('Test accuracy:', acc)

#classes = model.predict_classes(x[train_num:])
#acc2 = np_utils.accuracy(classes, y[train_num:])
#print('Test accuracy:', acc2)

def predict_one(s): #单个句子的预测函数
    s = np.array(doc2num(list(jieba.cut(s)), maxlen))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]
#------------------------------------------------
import json
import h5py
from keras.models import model_from_json
json_string = model.to_json()
ModelPath = 'model0813.json'
fd = open(ModelPath, 'w')
fd.write(json_string)
fd.close()
model.save_weights('model0813.h5')#80p = test accuracy
