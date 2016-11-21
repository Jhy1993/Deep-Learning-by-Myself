# -*- coding: utf-8 -*-
"""
Created on  Aug 22 15:03:08 2016

@author: Jhy_Bistu
"""
from train0822_v6 import DL_model
DL = DL_model()
data_path = 'comments_2016-08-08_v4.xls'
comment, label = DL.get_data(data_path)
content = DL.preprocess1(comment, label)
min_count = 5
content, word_index = DL.get_word_index(content, min_count)
MAX_SEQUENCE_LENGTH = 100
content['doc2num'] = content['c'].apply(lambda s: doc2num(s, MAX_SEQUENCE_LENGTH))
content = my_shuffle(content)
print('shuffle is ok!!!')
x = np.array(list(content['doc2num']))
y = np.array(list(content['label']))
print ('preprocess data(2/2) is ok!!!')
print y[11]
word2vec_path = 'wiki.zh.text.vector'
embeddings_matrix = DL.load_word2vec(word2vec_path)
print embeddings_matrix[11]
nb = 3
model = DL.get_model(x, y, embeddings_matrix, nb)
ModelPath = 'model_3c_0819.json'
WeightPath = 'model_3c_0819.h5'
DL.save_model(ModelPath, WeightPath)