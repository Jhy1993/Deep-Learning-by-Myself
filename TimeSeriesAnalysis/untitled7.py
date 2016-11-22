# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 17:42:32 2016

@author: Jhy_BUPT
README:

INPUT:

OUTPUT:

REFERENCE:

"""
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
x = x.sort([1])
plt.plot(x[1], x[2])
#=====================插值=======================
#def ployinterp_column(s, n, k=5):  
#  y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))] #取数  
#  y = y[y.notnull()] #剔除空值  
#  return lagrange(y.index, list(y))(n) #插值并返回插值结果  
#  
##逐个元素判断是否需要插值  
#for i in data.columns:  
#  for j in range(len(data)):  
#    if (data[i].isnull())[j]: #如果为空即插值。  
#      data[i][j] = ployinterp_column(data[i], j)  
#  
#data.to_csv(outputfile) #输出结果，写入文件  
#  