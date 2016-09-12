'''
simple NN 2-500-2

'''
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X, Y = datasets.make_moons(200, noise=0.2)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmp=plt.cm.Spectral)

# Train the logistic rgeression classifier
clf = linear_model.LogisticRegressionCV()
clf.fit(X, Y)
 
# Plot the decision boundary
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")

# Neural network
num_examples = len(X)
input_dim = 2
output_dim = 2

epsilon = 0.01
reg_lambda = 0.01

#iniliaztion
W1 = np.random.randn(hidden_dim, input_dim) / np.sqrt(input_dim)
b1 = np.random.randn(hidden_dim, 1)
W2 = np.random.randn(output_dim, hidden_dim) / np.sqrt(hidden_dim)
b2 = np.random.randn(output_dim, 1)

#forward propagation
def forward_propagation(X):
    z1 = np.dot(W1, X) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(W2, a1) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs
#calculate loss
def calculate_loss(X, Y):
    for i in range(len(X)):
        Loss += Y[i].T * np.log(forward_propagation(X[i]))
    return Loss = -1/num_examples * Loss


#back propagation
delta3 = probs
delta3



def tanh_prime(s):
    s = 1 - s**2
    return s
