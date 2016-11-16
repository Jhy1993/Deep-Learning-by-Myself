# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 21:26:39 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:
深度学习与计算机视觉系列(7)_神经网络数据预处理，正则化与损失函数
Reference:
http://blog.csdn.net/han_xiaoyang/article/details/50451460
"""

import numpy as np
x = np.random.randn(10, 10)
print x
U, S, V = np.linalg.svd(x)
#坐标旋转
x_rot = np.dot(x, U)
# 降维旋转
k = 5
xrot_redcued = np.dot(x, U[:, :k])
# whitening
x_white = x_rot / np.sqrt(S + 1e-5)

#===================drop=============
drop_prob = 0.5
def train_step(X):
    H1 = np.maximum(0, np.dot(W1, X) + b1)
    U1 = np.random.rand(*H1.shape) < drop_prob
    H1 *= U1
    
    H2 = np.maximum(0, np.dot(W2, H1) + b2)
    U2 = np.random.rand(*H2.shape) < drop_prob
    H2 *= U2
    
    out = np.dot(W2, H2) + b3
    
def predict(X):
    H1 = np.maximum(0, np.dot(W1, X) + b1) * drop_prob
    H2 = np.maximum(0, np.dot(W2, H1) + b2) * drop_prob
    out = np.dot(W3, H2) + b3
""" 
Inverted Dropout的版本，把本该花在测试阶段的时间，转移到训练阶段，从而提高testing部分的速度
"""

p = 0.5 # dropout的概率，也就是保持一个神经元激活状态的概率

def train_step(X):
  # f3层神经网络前向计算
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = (np.random.rand(*H1.shape) < p) / p # 注意到这个dropout中我们除以p，做了一个inverted dropout
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = (np.random.rand(*H2.shape) < p) / p # 这个dropout中我们除以p，做了一个inverted dropout
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3

  # 反向传播: 计算梯度... (这里省略)
  # 参数更新... (这里省略)

def predict(X):
  # 直接前向计算，无需再乘以p
  H1 = np.maximum(0, np.dot(W1, X) + b1) 
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  out = np.dot(W3, H2) + b3













