# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:38:28 2016

@author: Jhy_BUPT

README:

INPUT:

OUTPUT:

REFERENCE:

"""
from __future__ import print_function

import os
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization

from datetime import datetime
from sklearn.preprocessing import scale, MinMaxScaler


jjjhy= MinMaxScaler().fit_transform(jhy)