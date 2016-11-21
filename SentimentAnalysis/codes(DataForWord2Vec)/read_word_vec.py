#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
#import gensim
#w2vmodel = gensim.models.Word2Vec.load("wiki.zh.text.model")
with open('wiki.zh.text.vector') as f:
    lines  = f.readlines()
    k = 1
    for line in lines:
        if k < 2:
            print line
            print len(line)
        k +=1


