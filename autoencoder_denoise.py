# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:20:02 2016

@author: Jhy_BISTU
INPUT:

OUTPUT:

REFERENCE:

"""
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, noise
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, ((len(x_train), 1, 28, 28)))
x_test = np.reshape(x_test, ((len(x_test), 1, 28, 28)))

noise_factor = 0.4
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

