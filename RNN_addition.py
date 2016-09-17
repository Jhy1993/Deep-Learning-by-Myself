'''
RNN : teach RNN to learn addition
such as  a+ b = c
'''
import numpy as np 
import copy
def sigmoid(x):
    x = 1 / (1+np.exp(x))
    return x

def sigmoid_prime(x):
    return x * (1-x)

#generate dataset
int2binary = {}
binary_dim = 8

largest_number = pow(2, binary_dim)
binary = np.unpackbits(np.array([range(largest_number)], dtype=np.uint8).T, axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]

learning_rate = 0.1
input_dim =2
hidden_dim = 16
output_dim = 1

U = 2 * np.random.random((hidden_dim, input_dim)) - 1
V = 2 * np.random.random((output_dim, hidden_dim)) - 1
W = 2 * np.random.random((hidden_dim, hidden_dim)) - 1

U_update = np.zeros_like(U)
V_update = np.zeros_like(V)
W_update = np.zeros_like(W)

for j in range(1000):
    a_int = np.random.randint(largest_number / 2)
    a = int2binary[a_int]
    b_int = np.random.randint(largest_number / 2)
    b = int2binary[b_int]
    c_int = a_int + b_int
    c = int2binary[c_int]

    c_hat = np.zeros_like(c)

    OverAllError = 0

    output_deltas = list()
    hidden_layer_values = list()
    hidden_layer_values.append(np.zeros(hidden_dim))
    #forward propagation and calculate error
    for position in range(binary_dim):
        # order from right to left
        X = np.array([[a[binary_dim - 1 - position], b[binary_dim - 1 - position]]])
        X = X.T      
        Y = np.array([c[binary_dim - 1 - position]]).T
        Y = Y.T        
        # hidden layer
        hidden_layer = sigmoid(np.dot(U, X) + 
                        np.dot(W, hidden_layer_values[-1])
                        )
        # output layer
        output_layer = sigmoid(np.dot(V, hidden_layer))
        # output error and loss
        output_error = Y - output_layer
        output_deltas.append(output_error * sigmoid_prime(output_layer))

        OverAllError += np.abs(output_error[0])
        #decode estimate so we can print it
        c_hat[binary_dim - 1 - position] = np.round(output_layer[0][0])
        # store hidden layer so we can use it in the next timestep
        hidden_layer_values.append(copy.deepcopy(hidden_layer))
    future_hidden_layer_delta = np.zeros(hidden_dim)
    # back propagation and update weights
    for position in range(binary_dim):
        X = np.array([[a[position], b[position]]])
        hidden_layer = hidden_layer_values[-position - 1]
        prev_hidden_layer = hidden_layer_values[-position-2]
        #error on output layer
        output_delta = output_deltas[-position-1]
        #error on hidden_layer(t) propagation from hidden_layer(t+1) and output_layer(t)
        hidden_delta = np.dot(W.T, future_hidden_layer_delta) + \
                    np.dot(V.T, output_delta) * sigmoid_prime(hidden_layer)
        #update d(loss)/dw(L) = a(L-1) * delta(L)
         V_update += np.dot(hidden_layer.T, output_delta)
        W_update += np.dot(prev_hidden_layer.T, hidden_delta)
        U_update += np.dot(X.T, hidden_delta)

        future_hidden_layer_delta = hidden_delta
    U += learning_rate * U_update
    V += learning_rate * V_update
    W += learning_rate * W_update

    V_update *= 0
    W_update *= 0
    U_update *= 0

    if j % 1000 == 0:
        print ('Error: %s' %OverAllError)
        print ('Pred: %s' %d)
        print ('True: %s' %c)
        out = 0
        # 2 --> 10 
        for idx, x in enumerate(reversed(d)):
            out += x * pow(2, idx)
        print ('%s + %s = %s' %(a_int, b_int, out))
        print ('------------------------')

















