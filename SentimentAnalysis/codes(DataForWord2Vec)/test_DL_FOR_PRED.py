#!/usr/bin/env python
# -*- coding: utf-8 -*-
from DL_FOR_PRED import DL_for_predict



MAX_SEQUENCE_LENGTH = 100 
word2vec_path = 'wiki.zh.text.vector'
ModelPath = 'model_3c_0822.json'
WeightPath = 'model_3c_0822.h5'
word_index_path = 'word_index.pkl'

DL = DL_for_predict(
            MAX_SEQUENCE_LENGTH=MAX_SEQUENCE_LENGTH,
            word2vec_path=word2vec_path,
            ModelPath=ModelPath,
            WeightPath=WeightPath,
            word_index_path=word_index_path)
s = '款式不错，就是有点大'
predict_label = DL.GO(s)

print s, '---label is ---', predict_label