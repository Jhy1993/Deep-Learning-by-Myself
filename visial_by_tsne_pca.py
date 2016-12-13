# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:54:06 2016

@author: Jhy_BUPT
README:
Visualising high-dimensional datasets using PCA and tSNE
INPUT:

OUTPUT:

REFERENCE:
https://golog.co/blog/article/Visualising_high-dimensional_datasets_using_PCA_and_tSNE
"""

import numpy as np
from sklearn import datasets

mnist = datasets.load_digits(n_class=10)
X = mnist.data / 255.0
y = mnist.target

print X.shape, y.shape