# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:
Recurrent Neural Network on real data
INPUT:

OUTPUT:

REFERENCE:
https://github.com/BinRoot/TensorFlow-Book/blob/master/ch10_rnn/Concept03_rnn_real_world.ipynb
"""
from __future__ import print_function

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


class SeriesPredictor:
    """docstring for SeriesPredictor"""
    def __init__(self, input_dim, seq_size, hidden_size):
        self.input_dim = input_dim
        self.seq_size = seq_size
        self.hidden_size = hidden_size

        self.W_out = tf.Variable(tf.random_normal([hidden_size, 1]), name="W_out")
        self.b_out = tf.Variable(tf.random_normal([1]), name="b_out")
        self.x = tf.placeholder(tf.float32, [None, seq_size, input_dim])
        self.y = tf.placeholder(tf.float32, [None, seq_size])

        self.cost = tf.reduce_mean(tf.square(self.model() - self.y))
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(self.cost)

        self.saver = tf.train.Saver()

    def model(self):
        cell = rnn_cell.BasicLSTMCell(self.hidden_size)
        outputs, states = rnn.dynamic_rnn(cell, self.x, dtype=tf.float32)
        num_example = tf.shape(self.x)[0]
        W_repeated = tf.tile(tf.expand_dims(self.W_out, 0), [num_example, 1, 1])
        # expand_dims: let w dim[4, 3] become dim[1, 4, 3]
        # tile: let w[1, 4, 3] become [num_example, 4, 3]
        out = tf.batch_matmul(outputs, W_repeated) + self.b_out
        out = tf.squeeze(out)
        # squeeze : remove all dims of size 1 from a shape of a tensor
        # [1, 2, 1, 3, 1] --> [2, 3]
        return out

    def train(self, train_x, train_y, test_x, test_y):
        with tf.Session() as sess:
            tf.get_variable_scope().reuse_variables()
            sess.run(tf.initialize_all_variables())
            max_patience = 3
            patience = max_patience
            min_test_err = float('inf')
            step = 0
            while patience > 0:
                _, train_err = sess.run([sess.train_op, sess.cost])
                if step % 20 == 0:
                    test_err = sess.run(self.cost, feed_dict={self.x: test_x, self.y: train_y})
                    print('steps:{}  trian error: {}  test error'.fromat(step, train_err, test_err))
                    if test_err < min_test_err:
                        min_test_err = test_err
                        patience = max_patience
                    else:
                        patience -= 1
                    step += 1
                save_path = self.saver.save(sess, 'model.ckpt')
                print('model save to {}'.fromat(save_path))


        pass





        

    def test(self, sess, test_x):
        tf.get_variable_scope().reuse_variables()
        self.saver.restore(sess, 'model.ckpt')
        output = sess.run(sess.model(), feed_dict={self.x: test_x})
        return output

def plot_results(train_x, predictions, actual, filename):
    plt.figure()
    num_train = len(train_x)
    plt.plot(list(range(num_train)), train_x, color='b', label='training data')
    plt.plot(list(range(num_train, num_train + len(predictions))), predictions, color='r', label='predicted')
    plt.plot(list(range(num_train, num_train + len(actual))), actual, color='g', label='test data')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

