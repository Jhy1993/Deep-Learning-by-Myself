# -*- coding: utf-8 -*-
"""
Date: 2017-08-17
@author: Jhy_Bistu
INFO: training deep learning model for kuanshi 3classes
Input: comments_2016-08-08_v4.xls
Output: The DL model(save as file, can be load)

Version: 1.5
Update:  Add some Tricks + validation_data
1.4 use early stop  + function optimization
1.3 Use GRU (instead LSTM)
1.2 DataBalance(make all classes have same number)
"""
#%%---------------------data for process----------------
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import pandas as pd
import jieba
import numpy as np
import pickle
import json
import h5py
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM, GRU
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop 
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm 
from keras.regularizers import l2, activity_l2
#========================load data===================================
content = pd.read_excel('comments_2016-08-08_v4.xls')
content = content.rename(columns={'kuanshi' : 'label'})
comment = content['c']
label = content['label']
# content = pd.concat([comment, label], axis=1)

print ('load data is ok!!!')
#=================preprocess data=============================
def preprocess1(comment, label):
    #data balance
    pos_index = []
    mid_index = []
    neg_index = []
    # print ('len(label) is %d' %len(label))
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
    return content
def getmin(pos_num, mid_num, neg_num):
    min_num = pos_num
    if min_num > mid_num:
        min_num = mid_num
    if min_num > neg_num:
        min_num = neg_num
    return min_num
content = preprocess1(comment, label)
print ('preprocess data(1/2) is ok!!!')

#=====================word index===================================================
min_count = 5
def get_word_index(content, min_count):   
    allwords = []
    content['words'] = content['c'].apply(lambda s: list(jieba.cut(s))) #调用结巴分词
    for i in content['words']:
        allwords.extend(i)
    word_index = pd.Series(allwords).value_counts()
    word_index = word_index[word_index >= min_count]
    word_index[:]= range(1, len(word_index)+1)
    word_index[''] = 0#下标必须从0开始
    return content, word_index

content, word_index = get_word_index(content, min_count)
#=============================doc2num=============================================
MAX_SEQUENCE_LENGTH = 100
def doc2num(sentence, MAX_SEQUENCE_LENGTH):
    sentence = [i for i in sentence if i in word_index.index]
    sentence = sentence[:MAX_SEQUENCE_LENGTH] + [''] * max(0, MAX_SEQUENCE_LENGTH - len(sentence))
    return list(word_index[sentence])

content['doc2num'] = content['c'].apply(lambda s: doc2num(s, MAX_SEQUENCE_LENGTH))

#================================shuffle and modify format==================================
def my_shuffle(content):
    idx = range(len(content))
    np.random.shuffle(idx)
    content = content.loc[idx]
    return content
content = my_shuffle(content)

x = np.array(list(content['doc2num']))
y = np.array(list(content['label']))
print ('preprocess data(2/2) is ok!!!')
#=============================== load word2vec===============================
def load_word2vec(word2vec_path):
    embeddings_index = {}
    with open(word2vec_path) as f:
        lines = f.readlines()
        for line in lines:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    EMBEDDING_DIM = len(coefs)
    print ('EMBEDDING_DIM is %s' % EMBEDDING_DIM)
    print ('find %s word vectors' % len(embeddings_index))    
    embeddings_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
    for  i in  word_index.values:
        word = word_index.index[i].encode('utf-8')
        embeddings_vector = embeddings_index.get(word)
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector
    return embeddings_matrix
word2vec_path = 'wiki.zh.text.vector'
embeddings_matrix= load_word2vec(word2vec_path)
#================================= keras===============================
print ('start keras...')

y = np_utils.to_categorical(y, 3)
#建立模型
def get_model(x, y, embeddings_matrix, nb):
    EMBEDDING_DIM = embeddings_matrix.shape[1]
    model = Sequential()
    # model.add(Embedding(len(abc), 256, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Embedding(len(word_index)+1,
                            EMBEDDING_DIM,
                            weights=[embeddings_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
    model.add(LSTM(128, 
        init='glorot_uniform',
        inner_init='orthogonal',
        forget_bias_init='one',
        #activation='tanh',
        inner_activation='hard_sigmoid',
        # W_regularizer=l2(0.01),#None,
        # U_regularizer=l2(0.01),#None,
        # b_regularizer=l2(0.01),#None,
        dropout_W=0.2, 
        dropout_U=0.2, 
        return_sequences=True,
        #W_constraint=maxnorm(2)
        ))

    model.add(LSTM(128, 
        init='glorot_uniform',
        inner_init='orthogonal',
        forget_bias_init='one',
        #activation='tanh',
        inner_activation='hard_sigmoid',
        # W_regularizer=l2(0.01),#None,
        # U_regularizer=l2(0.01),#None,
        # b_regularizer=l2(0.01),#None,
        dropout_W=0.2, 
        dropout_U=0.2, 
        return_sequences=False,
        #W_constraint=maxnorm(2)
        ))
    # model.add(Dropout(0.2))


    model.add(Dense(3))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.002, decay=1e-6, 
        momentum=0.9, nesterov=True,
        clipnorm = 15)
    #rmsprop = RMSprop(lr = 0.001)
    model.compile(loss='categorical_crossentropy',
              #class_mode = "categorical",
              optimizer=sgd, #'rmsprop',#sgd',
              metrics=['accuracy'])
    early_Stop = EarlyStopping(
        monitor='acc',#'val_loss', 
        verbose=2,
        mode='auto',
        patience=10)
    batch_size = 32
    train_num = int(len(x) * 0.8)
    model.fit(x[:train_num], y[:train_num], 
               batch_size = batch_size, 
                shuffle=True, 
                verbose = 2,
                #validation_split=0.1,
                #validation_data=(x[train_num:], y[train_num:]),
                #callbacks=[early_Stop],
              nb_epoch=nb)
    print ('model training is ok!!!')
    #测试模型
    score, acc = model.evaluate(x[train_num:], y[train_num:], 
                                batch_size = batch_size)
    print('Test accuracy:', acc)
    return model
nb = 30
model = get_model(x, y, embeddings_matrix, nb)

#=======================save word_index===============================
output=open('word_index.pkl','wb')
pickle.dump(word_index,output)
print ('save word_index is ok!!')
#======================= save model==================================

def save_model(ModelPath, WeightPath):
    print ('start save model...')
    json_string = model.to_json()
    fd = open(ModelPath, 'w')
    fd.write(json_string)
    fd.close()
    model.save_weights(WeightPath)#80p = test accuracy
    print ('save model , ok!!!')
ModelPath = 'model_3c_0823.json'
WeightPath = 'model_3c_0823.h5'
save_model(ModelPath, WeightPath)
