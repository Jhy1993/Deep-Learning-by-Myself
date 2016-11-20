# -*- coding: utf-8 -*-
"""
Created on Fri Sep 02 11:09:26 2016

@author: box1
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, RepeatVector
from keras.layers import LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from util import categorical_accuracy_per_sequence

def build_CNN_LSTM(channels, width, height, lstm_output_size, nb_classes):
	model = Sequential()
	# 1 conv
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', 
		input_shape=(channels, height, width)))
	model.add(BatchNormalization(mode=0, axis=1))
	# 2 conv
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
	model.add(BatchNormalization(mode=0, axis=1))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
	# 3 conv
	model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
	model.add(BatchNormalization(mode=0, axis=1))
	# 4 conv
	model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
	model.add(BatchNormalization(mode=0, axis=1))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
	# flaten
	a = model.add(Flatten())
	# 1 dense
	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	# 2 dense
	model.add(Dense(512, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	# lstm
	model.add(RepeatVector(lstm_output_size))
	model.add(LSTM(512, return_sequences=True))
	model.add(TimeDistributed(Dropout(0.5)))
	model.add(TimeDistributed(Dense(nb_classes, activation='softmax')))
	model.summary()
	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=[categorical_accuracy_per_sequence],
				  sample_weight_mode='temporal'
				  )

	return model