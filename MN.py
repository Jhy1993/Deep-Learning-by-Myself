'''
Michael Nielsen

Version：1.0 Simple Neural Network
'''

import cPickle
import gzip
#======================== Load the MNIST data=========================================
def load_data():
    
    f = gzip.open('C:\\Users\\Jhy1993\\Desktop\\Deep-Learning-by-Myself\\data\\mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

a,b,c = load_data()
'''
#=======================================N N==============================================
class Network(object):
    """docstring for Network"""
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for y, x in sizes[1:], sizes[:-1]]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] 
                                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}:{1} / {2}".format(j, self.evaluate(test_data), n_test) 
            else:
                print "Epoch {0}".format(j)

    def update_mini_batch(self, mini_batch, eta):
        #update w, b use mini batch data
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = nabla_w + delta_nabla_w
            nabla_b = nabla_b + delta_nabla_b
        self.weights -= eta / len(mini_batch) * nabla_w
        self.biases  -= eta / len(mini_batch) * nabla_b

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        #feedforward
        activation = x
        activations.append(x)
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        #back propagation
        delta[-1] = self.mse_cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for L in xrange(2, self.num_layers):
            delta[-L] = np.dot(self.weights[-L+1].transpose(), delta) * sigmoid_prime(zs[-L])
            nabla_b[-L] = delta[-L]
            nabla_w[-L] = np.dot(delta[-L], activations[-L-1].transpose())
        return nabla_w, nabla_b

    def mse_cost_derivative(a, y):
        # 1/2 * (a-y)^2
        return (a - y)

    def feedforward(self, a):
        #input -> output
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        # return the num of NN predict correctly
        result_contrast = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in result_contrast)
        

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
'''