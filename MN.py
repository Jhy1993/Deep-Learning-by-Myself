'''
Michael Nielsen
NN
'''
class Network(object):
    """docstring for Network"""
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for y, x in sizes[1:], sizes[:-1]]

    def feedforward(self, a):
        #input -> output
        for w, b in zip(self.weights, self.biases)
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] 
                                for k in xrange(0,???????????? )]
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
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_w = [nb + dnb for nb, dnb in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nw + dnw for nw, dnw in zip(nabla_b, delta_nabla_b)]
            self.weights = [w - (eta / len(mini_batch))]

        pass


        pass

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
