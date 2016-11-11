# -*- coding: utf-8 -*-
"""
Created on Sat Oct 08 19:52:35 2016

@author: Jhy1993
Github: https://github.com/Jhy1993

README:

Reference:

"""

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
    













