# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:44 2016

@author: Jhy_BUPT
README:
Processing timeseries data
INPUT:

OUTPUT:

REFERENCE:
https://github.com/BinRoot/TensorFlow-Book/blob/master/ch10_rnn/Concept01_timeseries_data.ipynb
"""
from __future__ import print_function

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

def load_series(filename, series_index=1):
    try:
        with open(filename, 'w') as csvfile:
            csvreader = csv.reader(csvfile)
            data = [float(row[series_index])
                    for row in csvreader if len(row) > 1]
            normalized_data = (data - np.mean(data)) / np.std(data)
        return normalized_data
    except IOError:
        return None

def split_data(data, precent_train=0.8):
    num_row = len(data)
    train_data, test_data =[], []
    for idx, row in enumerate(data):
        if idx < int(num_row * precent_train):
            train_data.append(row)
        else:
            test_data.append(row)
    return train_data, test_data
    
if __name__ == '__main__':
    timeseries = load_series('sssssss.csv')
    
            
            


















