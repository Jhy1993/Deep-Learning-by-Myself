# -*- coding: utf-8 -*-
"""
Date: 2017-08-17
@author: Jhy_Bistu
INFO: training deep learning model for kuanshi 3classes
Input: comments_2016-08-08_v4.xls
Output: The DL model(save as file, can be load)
Version: 1.2
Update: DataBalance(make all classes have same number)
"""
#%%---------------------data for process----------------
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import pandas as pd
import jieba
import numpy as np
content = pd.read_excel('comments_2016-08-08_v4.xls')
comment = content['c']
label = content['kuanshi']
content = pd.concat([comment, label], axis=1)
#label = label.rename(columns={'kuanshi' : 'label'})
#content = content.rename(columns={'kuanshi' : 'label'})
print ('load data is ok!!!')
#=================预处理数据=============================
pos_index = []
mid_index = []
neg_index = []

for i in range(len(label)):
    if label[i] == 1:
        pos_index.append(label.index.get_loc(i))
    if label[i] == 0:
        mid_index.append(label.index.get_loc(i))
    if label[i] == -1:
        neg_index.append(label.index.get_loc(i))
pos_num = len(pos_index)
mid_num = len(mid_index)
neg_num = len(neg_index)
def getmin(pos_num, mid_num, neg_num):
    min_num = pos_num
    if min_num > mid_num:
        min_num = mid_num
    if min_num > neg_num:
        min_num = neg_num
    return min_num
min_num = getmin(pos_num, mid_num, neg_num)  
if min_num < 500:
    print ('Data is too little, we need more data!!!!!!!!!')
pos = comment[pos_index[:min_num]]
mid = comment[mid_index[:min_num]]
neg = comment[neg_index[:min_num]]

pos_label = label[pos_index[:min_num]]
mid_label = label[mid_index[:min_num]]
neg_label = label[neg_index[:min_num]]

comment_new = pd.concat([pos, neg, mid], ignore_index=True)
label_new =  pd.concat([pos_label, neg_label, mid_label], ignore_index=True)

content = pd.concat([comment_new, label_new], axis=1)
content = content.rename(columns={'kuanshi' : 'label'})
print ('preprocess data(1/2) is ok!!!')
#=====================分词===================
allwords = []
content['words'] = content['c'].apply(lambda s: list(jieba.cut(s))) #调用结巴分词
for i in content['words']:
    allwords.extend(i)
min_count = 5
MAX_SEQUENCE_LENGTH = 100
word_index = pd.Series(allwords).value_counts()
word_index = word_index[word_index >= min_count]
word_index[:]= range(1, len(word_index)+1)
word_index[''] = 0#下标必须从0开始

def doc2num(sentence, MAX_SEQUENCE_LENGTH):
    sentence = [i for i in sentence if i in word_index.index]
    sentence = sentence[:MAX_SEQUENCE_LENGTH] + [''] * max(0, MAX_SEQUENCE_LENGTH - len(sentence))
    return list(word_index[sentence])

content['doc2num'] = content['c'].apply(lambda s: doc2num(s, MAX_SEQUENCE_LENGTH))

#shuffle
idx = range(len(content))
np.random.shuffle(idx)
content = content.loc[idx]

#translate data into keras format
x = np.array(list(content['doc2num']))
y = np.array(list(content['label']))
print ('preprocess data(2/2) is ok!!!')
#%%---------------------------1 load word2vec------------------
embeddings_index = {}
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

#%%----------------------------2 keras-------------------
print ('start keras...')
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop 
y = np_utils.to_categorical(y, 3)
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


model.add(Dense(3))
model.add(Activation('softmax'))
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#rmsprop = RMSprop(lr = 0.001)
model.compile(loss='categorical_crossentropy',
              #class_mode = "categorical",
              optimizer=sgd, #'rmsprop',#sgd',
              metrics=['accuracy'])

 
batch_size = 32
train_num = int(len(x) * 0.8)
 
model.fit(x[:train_num], y[:train_num], 
  batch_size = batch_size, 
  shuffle=True, 
  nb_epoch=3)
 
score, acc = model.evaluate(x[train_num:], y[train_num:], 
  batch_size = batch_size)
print('Test accuracy:', acc)
#%%---------------------------3 predict one ------------

def predict_one(s): #单个句子的预测函数
   s = np.array(doc2num(list(jieba.cut(s)), MAX_SEQUENCE_LENGTH))
   s = s.reshape((1, s.shape[0]))
   return model.predict_classes(s)
s1 = '款式不错，就是有点大'
predict_label1 = predict_one(s1)
print s1, '---label is ---', predict_label1

s2 = '显胖，版型不好，不喜欢'
predict_label2 = predict_one(s2)
print s2, '---label is ---', predict_label2

s3 = '一般吧'
predict_label3 = predict_one(s3)
print s3, '---label is ---', predict_label3
#----------------save word_index--------------------------
import pickle
output=open('word_index.pkl','wb')
pickle.dump(word_index,output)
print ('save word_index is ok!!')
#%%------------------------4 save model----------------------
import json
import h5py
from keras.models import model_from_json
print ('start save model...')
json_string = model.to_json()
ModelPath = 'model_3c_0813.json'
fd = open(ModelPath, 'w')
fd.write(json_string)
fd.close()
model.save_weights('model_3c_0813.h5')#80p = test accuracy
print ('save model , ok!!!')
