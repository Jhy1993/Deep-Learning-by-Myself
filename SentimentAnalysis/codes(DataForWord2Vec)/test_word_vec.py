#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import gensim
model = gensim.models.Word2Vec.load("wiki.zh.text.model")
y = model.most_similar(u"款式", topn=20)
print ('与款式最相似的词（20个）')
for i in y:
    print i[0], i[1]
