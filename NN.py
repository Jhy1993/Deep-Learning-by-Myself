'''
simple NN 2-500-2
Reference: http://python.jobbole.com/82208/
'''
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegressionCV
import numpy as np
import matplotlib.pyplot as plt
#=====================show Data===========================
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.2)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()


def plot_decision_boundary(pred_func):
    #设定最大最小值, 附加一点边缘填充
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #画图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

#=====================================LogisticRegressionCV===========
clf = LogisticRegressionCV()
clf.fit(X, y)
plot_decision_boundary(lambda x: clf.predict(x))
plt.title('LogisticRegressionCV')
plt.show()
#==============================Neural Network===========================
# Neural network
num_examples = len(X)
nn_input_dim = 2
nn_hidden_dim = 500
nn_output_dim = 2

epsilon = 0.01
reg_lambda = 0.01

#iniliaztion
W1 = np.random.randn(nn_hidden_dim, nn_input_dim) / np.sqrt(nn_input_dim)
b1 = np.random.randn(nn_hidden_dim, 1)
W2 = np.random.randn(nn_output_dim, nn_hidden_dim) / np.sqrt(nn_hidden_dim)
b2 = np.random.randn(nn_output_dim, 1)

#forward propagation
def forward_propagation(X):
    z1 = np.dot(W1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(W2, a1) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs
#calculate loss
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model
    z1 = np.dot(X, W1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1, W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    #calculate loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(calculate_loss)
    # regular
    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def build_model(nn_hidden_dim, num_passes=2000, print_loss=False):
    #random initial
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hidden_dim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hidden_dim))
    W2 = np.random.randn(nn_hidden_dim, nn_output_dim) / np.sqrt(nn_hidden_dim)
    b2 = np.zeros((1, nn_output_dim))
    # model we will learn
    model = {}
    #start SGD
    for i in xrange(0, num_passes):
        #forward propagation
        z1 = np.dot(X, W1) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        #back propagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = np.dot(a1.T, delta3)
        db2 = delta3
        delta2 = np.dot(delta3, W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = delta2
        #Add Regular
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1 
        #Update parameter
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2    
        b2 += -epsilon * db2
        # Get real model with W, b
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        #Whether show 
        if print_loss and i % 1000 ==0:
            print('Loss after iter %i: %f' %(i, calculate_loss(model)))
    return model

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    #forward propagation
    z1 = np.dot(x, W1) + b1
    a1 = np.tanh(z1)    
    z2 = np.dot(a1, W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

model = build_model(3, print_loss=True)

plot_decision_boundary(lambda x: predict(model, x))
plt.title('NN decision boundary')
plt.show()

# Test for different size of hidden layer
plt.figure(figsize=(15, 32))
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]

for i, nn_hidden_dim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer size %d' %(nn_hidden_dim))
    model = build_model(nn_hidden_dim)
    plot_decision_boundary(lambda x: predict(model, x))
plt.show()




