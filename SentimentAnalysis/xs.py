# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 16:02:37 2016

@author: Jhy
"""

import numpy as np
import pandas as pd
import jieba
cw = lambda x: list(jieba.cut(x)) 
com = pd.read_excel('comments_2016-08-08_v2.xlsx')#, header=['评价内容'])
com = com[com['comment'].notnull()]
com['words'] = com['comment'].apply(cw)
data = list(com['comment'])
label = list(com['kuanshi'])
#for i in range(len(label)):
#    print i
#    if label[i-1] == 0:
#        del (label[i-1])
#        del (data[i-1])
l1 = []
l2 = []
for id, num in enumerate(label):
    if num == 1:
        l1.append(id)
    if num == -1:
        l2.append(id)
pos = [data[i] for i in l1]    
neg = [data[i] for i in l2]
   
for i in pos:
    i = jieba.cut(i)
#    print id, num


        
#fenci = lambda x: list(jieba.cut(x))
#com['words'] = com['comment'].apply(fenci)

'''
maxlen = 100 #截断词数
min_count = 5 #出现次数少于该值的词扔掉。这是最简单的降维方法
 
content = []
for i in all_['words']:
	content.extend(i)
 
abc = pd.Series(content).value_counts()
abc = abc[abc >= min_count]
abc[:] = range(1, len(abc)+1)
abc[''] = 0 #添加空字符串用来补全
 
def doc2num(s, maxlen): 
    s = [i for i in s if i in abc.index]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    return list(abc[s])
 
all_['doc2num'] = all_[0].apply(lambda s: doc2num(s, maxlen))
 
#手动打乱数据
idx = range(len(all_))
np.random.shuffle(idx)
all_ = all_.loc[idx]
 
#按keras的输入要求来生成数据
x = np.array(list(all_['doc2num']))
y = np.array(list(all_['label']))
y = y.reshape((-1,1)) #调整标签形状
 

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
 
#建立模型
model = Sequential()
model.add(Embedding(len(abc), 256, input_length=maxlen))
model.add(LSTM(128)) 
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
batch_size = 128
train_num = 15000
 
model.fit(x[:train_num], y[:train_num], batch_size = batch_size, nb_epoch=30)
 
model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)
 
def predict_one(s): #单个句子的预测函数
    s = np.array(doc2num(list(jieba.cut(s)), maxlen))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]
'''