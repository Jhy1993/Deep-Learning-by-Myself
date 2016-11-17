# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:51:04 2016

@author: Jhy_BUPT
INPUT:

OUTPUT:

REFERENCE:

"""
import numpy as np
import time
t = time.time()
r = np.sin(t)
print('radnom is {}'.format(r))
a = 1
b = 10


def J(a):
    b = a * 2
    return b
J(a)
