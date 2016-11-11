# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 22:03:23 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:
深度学习与计算机视觉系列(9)_
串一串神经网络之动手实现小例子
Reference:
http://blog.csdn.net/han_xiaoyang/article/details/50521072
"""
import numpy as np
import matplotlib.pyplot as plt
N = 100 # 每个类中的样本点
D = 2 # 维度
K = 3 # 类别个数
X = np.zeros((N*K,D)) # 样本input
y = np.zeros(N*K, dtype='uint8') # 类别标签
for j in xrange(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# 可视化一下我们的样本点
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

#===========linear + softmax + crossentropy=========
num_examples = X.shape[0]
reg = 1e-3
step_size = 1e-0

W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))


for i in xrange(200):
    scores = np.dot(X, W) + b 
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    corect_loss = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_loss) / num_examples
    reg_loss = 0.5 * reg * np.sum(W * W)
    loss = data_loss + reg_loss
    if i % 10 == 0:
        print('iter {}, loss {}'.format(i, loss))
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)
    
    dW += reg * W
    
    W += -step_size * dW
    b += -step_size * db
    
scores = np.dot(X, W) + b 
pred = np.argmax(scores, axis=1)
print('training acc: {}'.format(np.mean(pred == y)))
    

#===============NN==============
h = 100
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

step_size = 1e-0
reg = 1e-3

num_examples = X.shape[0]
for i in xrange(10000):
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
                                #N*s1 s1*s2   +  s2
    scores = np.dot(hidden_layer, W2) + b2
                    #N * s2 s2 * s3   + s3
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss
    if i % 100 == 0:
        print('iter: {}, loss; {}'.format(i, loss))

    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    dhidden = np.dot(dscores, W2.T)
    dhidden[hidden_layer <= 0] = 0

    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)
    dW += reg * W
    dW2 += reg * W2

    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2

hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2

pred = np.argmax(scores, axis=1)
print('training accuracy: {}'.format(np.mean(pred == y)))









