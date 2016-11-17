# -*- coding: utf-8 -*-
"""
Created on Wed Jul 06 10:52:55 2016

@author: Jhy
"""
import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
import pandas as pd
df = pd.read_csv('C:\Users\Jhy\Desktop\c1.txt',
                 sep='^', header=None, dtype=str, na_filter=False)
fenci=lambda x:list(jieba.cut(x))
#def fenci(x):
#    fenci=jieba.cut(x)
#    return fenci
def jia(x):
    xx=round(float(x))   
    y=xx+100
    return y
df.columns=['num','user','num2','name','PJ','comment']
df['comment']=df['comment'].apply(fenci)
df['num']=df['num'].apply(jia)