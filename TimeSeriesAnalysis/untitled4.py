# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:

INPUT:

OUTPUT:

REFERENCE:

"""
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np
import pandas as pd
a= [11, 2, 5, 3, 2, 0]
b = [1,2 , 3, 4, 5, 6]
x = []
y = []
for i in range(len(a)):
    for j in range(0, 2):
        x.append(a[j])
    for k in range(-2, 0):
        print(k)
        y.append(b[k])
        
#print(x)
#print(y)
for k in range(-2):
    print(k)