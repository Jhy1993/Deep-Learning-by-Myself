#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
0 load data
1 load wiki word2vec
2 LSTM training
3 predict one

2分类： 1 好评
        0 差评
'''
#%%----------------0 load data----------------------
import numpy as np
import pandas as pd
import jieba
import re

MAX_SEQUENCE_LENGTH = 100 
min_count = 5

pos = pd.read_excel('pos3.xls', header=None)
pos['label'] = 1
neg = pd.read_excel('neg3.xls', header=None)
neg['label'] = 0
all = pos.append(neg, ignore_index=True)
all['words'] = all[0].apply(lambda s: list(jieba.cut(s))) #调用结巴分词
 
content = []
for i in all['words']:
    content.extend(i)# 有重复


abc = pd.Series(content).value_counts()
abc = abc[abc >= min_count]
abc[:] = range(1, len(abc)+1)
abc[''] = 0 #添加空字符串用来补全
word_index = abc
def doc2num(s, MAX_SEQUENCE_LENGTH): 
    s = [i for i in s if i in word_index.index]
    s = s[:MAX_SEQUENCE_LENGTH] + ['']*max(0, MAX_SEQUENCE_LENGTH-len(s))
    return list(word_index[s])
 
all['doc2num'] = all[0].apply(lambda s: doc2num(s, MAX_SEQUENCE_LENGTH))

#手动打乱数据
idx = range(len(all))
np.random.shuffle(idx)
all = all.loc[idx]
 
#按keras的输入要求来生成数据
x = np.array(list(all['doc2num']))
y = np.array(list(all['label']))
y = y.reshape((-1,1)) #调整标签形状
#%%---------------------------1 load word2vec------------------
embeddings_index = {}
#with open('wiki.zh.text.vector') as f:
#    lines = f.readlines()
#    kk = 1
#    for line in lines:
#        if kk <1000:
#            values = line.split()
#            word = values[0]
#            coefs = np.asarray(values[1:], dtype='float32')
#            embeddings_index[word] = coefs
#        kk +=1
with open('wiki.zh.text.vector') as f:
    lines = f.readlines()
    for line in lines:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

#print embeddings_index['的']
EMBEDDING_DIM = len(coefs)
print ('EMBEDDING_DIM is %s' % EMBEDDING_DIM)
print ('find %s word vectors' % len(embeddings_index))    


#word_index = embeddings_index
embeddings_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))

for  i in  word_index.values:
    word = word_index.index[i].encode('utf-8')
    embeddings_vector = embeddings_index.get(word)
    if embeddings_vector is not None:
        embeddings_matrix[i] = embeddings_vector

#for word, wordvec in word_index.items():
#    embeddings_vector = embeddings_index.get(word)
#    if embeddings_vector is not None:
#        embeddings_matrix[']

#%%----------------------------2 keras-------------------
print ('start keras...')
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop 
#embedding_layer = Embedding(len(word_index)+1,
#                            EMBEDDING_DIM,
#                            weights=[embeddings_matrix],
#                            input_length=MAX_SEQUENCE_LENGTH,
#                            trainable=False)
#建立模型
model = Sequential()
# model.add(Embedding(len(abc), 256, input_length=MAX_SEQUENCE_LENGTH))
model.add(Embedding(len(word_index)+1,
                            EMBEDDING_DIM,
                            weights=[embeddings_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
model.add(LSTM(128,return_sequences=True)) 
#model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))


model.add(Dense(1))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#rmsprop = RMSprop(lr = 0.001)
model.compile(loss='binary_crossentropy',
              optimizer=sgd, #'rmsprop',#sgd',
              metrics=['accuracy'])

 
batch_size = 32
train_num = int(len(x) * 0.8)
 
model.fit(x[:train_num], y[:train_num], batch_size = batch_size, shuffle=True, nb_epoch=20)
 
score, acc = model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)
print('Test accuracy:', acc)
#---------------------------3 predict one ------------

def predict_one(s): #单个句子的预测函数
   s = np.array(doc2num(list(jieba.cut(s)), MAX_SEQUENCE_LENGTH))
   s = s.reshape((1, s.shape[0]))
   return model.predict_classes(s)
s1 = '款式不错，就是有点大'
predict_label1 = predict_one(s1)
print s1, '---label is ---', predict_label1

s2 = '质量很差，不喜欢'
predict_label2 = predict_one(s2)
print s2, '---label is ---', predict_label2