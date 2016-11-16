'''
RNN 
1. Initial 
2. forward_propagation
3. calculate_total_loss -> calculate_loss
4. bptt
5. gradient_check
6. numpy_sgd_step
7. train_with_sgd
utils:
1. tanh_prime
'''
import numpy as np
class RNNNumpy:
    """docstring for RNNNumpy"""
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        T = len(x)
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        #=============????????????????????????
        #output-o
        o = np.zeros((T, self.word_dim))
        for i in np.arange(T):
            s[t] = np.tanh(np.dot(self.U, x[t]) + np.dot(self.W, s[t-1]))
            o[t] = softmax(np.dot(self.V, s[t]))
        return [o, s]

    def predict(self, x):
        o, s = self.forward_propagation(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        for i in np.arange(len(y)):
            o, s = self.forward_propagation(x[i])
            correct_predict = o[np.arange(len(y[i])), y[i]]
#===========????????????????
            L += -1 * np.sum(np.log(correct_predict))
        return L

    def calculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        return self.calculate_total_loss(x, y) / N

    def bptt(self, x, y):
        T = len(y)
        o, s = self.forward_propagation(x)
        # gradient
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        for t in np.arange(T)[::-1]:
            dLdV +=np.outer(delta_o[t], s[t].T)
            delta_t = np.dot(self.V.T, delta_o[t]) * tanh_prime(s[t])
            for bptt_step in np.arange(max(0, t-bptt_truncate), t+1)[::-1]:
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:, x[bptt_step]] += delta_t
                delta_t = np.dot(self.W.T, delta_t) * tanh_prime(s[bptt_step-1])
        return [dLdU, dLdV, dLdW]

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        bptt_gradients = self.bptt(x, y)
        model_parameters = ['U', 'V', 'W']
        for pidx, pname in enumerate(model_parameters):
            print pidx, pname
        pass

    def numpy_sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW

    def train_with_sgd(model, X_train, y_train, 
                    learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
        losses = []
        num_examples_seen = 0
        for epoch in range(nepoch):
            for i in range(len(y_train)):
                model.sgd_step(X_train[i], y_train[i], learning_rate)
                num_examples_seen += 1
        pass
def tanh_prime(s):
    s_p = 1 - s**2
    return s_p

        
