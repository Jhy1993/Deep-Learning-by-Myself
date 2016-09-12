'''
 mysef
RNN part 23
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
        # grandient
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        for t in np.arange(T)[::-1]:
            dLdV +=np.outer(delta_o[t], s[t].T)
            delta_t = np.dot(self.V.T, delta_o[t]) * tanh_prime(s[t])
            for bptt_step in np.arange(max(0, t-bptt_truncate), t+1)[::-1]:
        pass
        
def tanh_prime(s):
    s_p = 1 - s**2
    return s_p

        
