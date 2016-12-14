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
x = np.array([[1, 1, 3], [4,5,9]])
mm = np.max(x[:,2])
print(mm)
print(x)
for i in range(x.shape[1]):
    m = np.max(x[:,i])
    x[:,i] /= m
print(x)