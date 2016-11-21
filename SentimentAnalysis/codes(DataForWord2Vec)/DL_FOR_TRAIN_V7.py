# -*- coding: utf-8 -*-
"""
Date: 2017-08-22
@author: Jhy_Bistu
INFO: training deep learning model for kuanshi 3classes
Input: comments_2016-08-08_v4.xls
Output: The DL model(save as file, can be load)

Version: 1.7
Update:  Add regular + maxnorm
1.6 Package into class + BatchNormalization
1.5 Add some Tricks + validation_data(cancel)
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
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop 
from keras.callbacks import EarlyStopping
from keras.constraints import maxnorm 
from keras.regularizers import l2, activity_l2
#========================load data===================================
class DL_for_train:
    """docstring for DL_model"""
    def __init__(self,
                data_path = 'comments_2016-08-08_v4.xls',
                MAX_SEQUENCE_LENGTH = 100,
                word2vec_path = 'wiki.zh.text.vector',
                nb = 80,
                min_count = 5,
                ModelPath = 'model_3c_0823.json',
                WeightPath = 'model_3c_0823.h5',
                word_index_path = 'word_index.pkl'
                ):
        self.data_path = data_path#'comments_2016-08-08_v4.xls'
        self.MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH#100
        self.word2vec_path = word2vec_path#'wiki.zh.text.vector'
        self.nb = nb#30
        self.min_count = min_count#5
        self.ModelPath = ModelPath#'model_3c_0822.json'
        self.WeightPath = WeightPath#'model_3c_0822.h5'
        self.word_index_path = word_index_path#'word_index.pkl'
    def get_data(self):
        content = pd.read_excel(self.data_path)
        content = content.rename(columns={'kuanshi' : 'label'})
        comment = content['c']
        label = content['label']
        print ('load data is ok!!!')
        return comment, label

    def data_balance(self, comment, label):
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

        min_num = self.getmin(pos_num, mid_num, neg_num)  
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
        print ('data balance is ok!!!')
        return content

    def get_word_index(self, content):   
        allwords = []
        content['words'] = content['c'].apply(lambda s: list(jieba.cut(s))) 
        for i in content['words']:
            allwords.extend(i)
        word_index = pd.Series(allwords).value_counts()
        word_index = word_index[word_index >= self.min_count]
        word_index[:]= range(1, len(word_index)+1)
        word_index[''] = 0#下标必须从0开始
        print ('word_index is ok!!!')
        return content, word_index

    def doc2num(self, sentence, word_index):
        sentence = [i for i in sentence if i in word_index.index]
        sentence = sentence[:self.MAX_SEQUENCE_LENGTH] + [''] * max(0, self.MAX_SEQUENCE_LENGTH - len(sentence))
        # print ('doc2num is ok!!!')
        return list(word_index[sentence])
      
    def get_doc2num(self, content, word_index):
        content['doc2num'] = content['c'].apply(lambda s: self.doc2num(s, word_index))
        print ('doc2num is ok!!!')
        return content

    def load_word2vec(self, word_index):
        embeddings_index = {}
        with open(self.word2vec_path) as f:
            lines = f.readlines()
            for line in lines:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        EMBEDDING_DIM = len(coefs)
        #print ('EMBEDDING_DIM is %s' % EMBEDDING_DIM)
        #print ('find %s word vectors' % len(embeddings_index))    
        embeddings_matrix = np.zeros((len(word_index)+1, EMBEDDING_DIM))
        for  i in  word_index.values:
            word = word_index.index[i].encode('utf-8')
            embeddings_vector = embeddings_index.get(word)
            if embeddings_vector is not None:
                embeddings_matrix[i] = embeddings_vector
        print('load embedding_matrix is ok!!!')
        return embeddings_matrix

    def get_model(self, x, y, embeddings_matrix, word_index):
        print ('start keras...')
        y = np_utils.to_categorical(y, 3)
        EMBEDDING_DIM = embeddings_matrix.shape[1]
        model = Sequential()
        model.add(Embedding(len(word_index)+1,
                            EMBEDDING_DIM,
                            weights=[embeddings_matrix],
                            input_length=self.MAX_SEQUENCE_LENGTH,
                            trainable=False))
        model.add(LSTM(256, 
                #init='glorot_uniform',
                #inner_init='orthogonal',
                #forget_bias_init='one',
                #activation='relu',
                #inner_activation='hard_sigmoid',
                W_regularizer=l2(0.01),
                U_regularizer=None,
                b_regularizer=None,
                #dropout_W=0.2,
                #dropout_U=0.2,
                return_sequences=True))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(epsilon=1e-04))

        model.add(LSTM(128, 
                #init='glorot_uniform',
                #inner_init='orthogonal',
                #forget_bias_init='one',
                #activation='relu',
                #inner_activation='hard_sigmoid',
                #W_regularizer=None,
                #U_regularizer=None,
                #b_regularizer=None,
                #dropout_W=0.2,
                #dropout_U=0.2,
                return_sequences=False))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(epsilon=1e-04))

        model.add(Dense(3,
                    W_constraint=maxnorm(15)
                    ))
        model.add(Activation('softmax'))
        sgd = SGD(lr=0.001, 
                    decay=1e-6, 
                    momentum=0.9, 
                    nesterov=True
                    #clipnorm = 15
                    )
        #rmsprop = RMSprop(clipnorm=0.1, epsilon=5e-04, lr = 0.001)
        #adam = Adam(epsilon=1e-03, clipnorm-0.1)
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
                validation_data=[x[train_num:], y[train_num:]],
                #callbacks=[early_Stop],
              nb_epoch=self.nb)
        print ('model training is ok!!!')
        #测试模型
        score, acc = model.evaluate(x[train_num:], y[train_num:], 
                                batch_size = batch_size)
        print('Test accuracy:', acc)
        return model

    def save_model(self, model):
        #print ('start save model...')
        json_string = model.to_json()
        fd = open(self.ModelPath, 'w')
        fd.write(json_string)
        fd.close()
        model.save_weights(self.WeightPath)#80p = test accuracy
        print ('save model , ok!!!')

    def get_data_format(self, content):
        x = np.array(list(content['doc2num']))
        y = np.array(list(content['label']))
        print ('data format(keras) is ok!!!')
        return x, y

    def my_shuffle(self, content):
        idx = range(len(content))
        np.random.shuffle(idx)
        content = content.loc[idx]
        return content

    def save_word_index(self, word_index):
        output=open(self.word_index_path,'wb')
        pickle.dump(word_index, output)
        print ('save word_index is ok!!')

    def GO(self):
        comment, label = self.get_data()
        content = self.data_balance(comment, label)

        content, word_index = self.get_word_index(content)
        content = self.get_doc2num(content, word_index)    
        content = self.my_shuffle(content)
        x, y = self.get_data_format(content)
    
        embeddings_matrix = self.load_word2vec(word_index)
        model = self.get_model(x, y, embeddings_matrix, word_index)
        self.save_model(model)
        return model
        
    def getmin(self, pos_num, mid_num, neg_num):
        min_num = pos_num
        if min_num > mid_num:
            min_num = mid_num
        if min_num > neg_num:
            min_num = neg_num
        return min_num



#==================test==============================
if __name__ == '__main__':
    DL = DL_for_train()
    model = DL.GO()
    
    # comment, label = DL.get_data()
    # content = DL.data_balance(comment, label)

    # content, word_index = DL.get_word_index(content)
    # content = DL.get_doc2num(content, word_index)    
    # content = DL.my_shuffle(content)
    # x, y = DL.get_data_format(content)
    
    # embeddings_matrix = DL.load_word2vec(word_index)
    # model = DL.get_model(x, y, embeddings_matrix, word_index)
    # ModelPath = 'model_3c_0822.json'
    # WeightPath = 'model_3c_0822.h5'
    # DL.save_model(ModelPath, WeightPath)

