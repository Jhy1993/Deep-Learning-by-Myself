# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 21:14:46 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:

Reference:

"""

import numpy as np
mat = np.array([[1.0, 2, 10], [3, 4, 5], [3, 5, 10]])
print mat
mat2 = mat - np.diag(np.diag(mat))
print mat2
mat3 = np.reciprocal(mat)
print mat3
print np.diag(np.sum(mat, axis=1))
print len(mat)
mat4 = np.array([[1.0, 2, 10], [3, 4, 5]])
print mat4.shape[0], mat4.shape[1]