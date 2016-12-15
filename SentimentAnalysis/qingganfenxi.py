# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 16:31:28 2016
python3

@author: Jhy
"""
from __future__ import absolute_import #导入3.x的特征函数
from __future__ import print_function
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
 
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
 
neg=pd.read_excel('neg.xls',header=None,index=None)
pos=pd.read_excel('pos.xls',header=None,index=None) #读取训练语料完毕
pos['mark']=1
neg['mark']=0 #给训练语料贴上标签
pn=pd.concat([pos,neg],ignore_index=True) #合并语料
#neglen=len(neg)
#poslen=len(pos) #计算语料数目

cw = lambda x: list(jieba.cut(x)) #定义分词函数
pn['words'] = pn[0].apply(cw)
comment = pd.read_excel('sum.xls') #读入评论内容
#comment = pd.read_csv('a.csv', encoding='utf-8')
comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw) #评论分词 
 
d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True) 
w = [] #将所有词语整合在一起
for i in d2v_train:
  w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
#del w,d2v_train
dict['id']=list(range(1,len(dict)+1))
 
get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent) #速度太慢
maxlen = 50# 词向量长度

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
#训练集
x_train = np.array(list(pn['sent']))[::2] 
y_train = np.array(list(pn['mark']))[::2]
#测试集
x_test = np.array(list(pn['sent']))[1::2] 
y_test = np.array(list(pn['mark']))[1::2]
x_train = np.array(list(pn['sent'])) #全集
y_train = np.array(list(pn['mark']))

'''
print('Build model...')

model = Sequential()
model.add(Embedding(len(dict)+1, 256))
model.add(LSTM(128,init='glorot_uniform')) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics= ['acc'])
 
model.fit(x_train, y_train, batch_size=16, nb_epoch=10,verbose=1) #训练时间为若干个小时
 
classes = model.predict_classes(x_train)
acc = np_utils.accuracy(classes, y_train)
print('Test accuracy:', acc)
score = model.evaluate(x_test, y_test, batch_size=16)
'''