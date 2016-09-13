import numpy as np 
'''
a = np.array([[1,2],[3,4]])
print  a[1]
exp_s = np.exp(a)
print exp_s

for i in xrange(0, 10, 3):
    print i
'''
x = np.array([[1,2,3], [1,2,4]])
y = np.array([[2,2,2]])
# for i, j in zip(x, y):
#     print i+j
#
# print x.transpose()
# print  x.T
# for i in xrange(2, 5):
#     print i
for (a, b) in zip(x[0,:], x[1,:]):
    print a, b
print sum(int(a == b) for (a, b) in zip(x[0,:], x[:,1]))
