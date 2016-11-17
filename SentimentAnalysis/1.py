# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import jieba
 
pos = pd.read_excel('pos3.xls', header=None)
pos['label'] = 1
neg = pd.read_excel('neg3.xls', header=None)
neg['label'] = 0
all = pos.append(neg, ignore_index=True)
all['words'] = all[0].apply(lambda s: list(jieba.cut(s))) #调用结巴分词

maxlen = 100 
min_count = 5
 
content = []
for i in all['words']:
    content.extend(i)

abc = pd.Series(content).value_counts()
abc = abc[abc >= min_count]
abc[:] = range(1, len(abc)+1)
abc[''] = 0 #添加空字符串用来补全