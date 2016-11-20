# -*- coding: utf-8 -*-
"""
Created on Thu Sep 01 17:26:06 2016

@author: box1
"""
import sys
import numpy as np
#import cv
from PIL import Image
import random
from io import BytesIO
#from captcha.image import ImageCaptcha
import os
from pylab import *

from keras.models import Sequential, Graph
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
class ALL_FOR_DATA(object):
    """docstring for ALL_FOR_DATA"""
    def __init__(self, arg):
        super(ALL_FOR_DATA, self).__init__()
        self.arg = arg
            
    def generate_captcha():
        for k in range(10):
            data = []
            label = []
            for i in range(4):
                pass
    def gen_rand():
        buf = ""
        for i in range(4):
            
            buf += str(random.randint(0,9))
            #print buf
        return buf

    def get_label(buf):
        a = [int(x) for x in buf]
        return np.array(a)

    def gen_sample(captcha, width, height):
        num = gen_rand()
        img = captcha.generate(num)
        img = np.fromstring(img.getvalue(), dtype='uint8')
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (width, height))
        img = np.multiply(img, 1/255.0)
        img = img.transpose(2, 0, 1)
        return (num, img)




class DL_FOR_Captcha:
    """docstring for ClassName"""
    def __init__(self, foldpath='C:\\Users\\Jhy1993\\Desktop\\picture_DL\\2',
                    nb_class=10,
                    row=40,
                    col=150):
        self.nb_class = nb_class
        self.foldpath = foldpath
        self.row = row
        self.col = col

    def load_data(self):
        #include data and label
        
        filelist = os.listdir(self.foldpath)
        num = len(filelist)
        data = np.empty((num, 1, self.row, self.col), dtype="float32")
        #label = np.empty((num,), dtype="uint8")
        label = []
        for i in range(num):
            #print file
            img = Image.open(os.path.join(self.foldpath, filelist[i])).convert('L')
            arr = np.asarray(img, dtype="float32")
            data[i, :, :, :] = arr
            
            label.append(filelist[i].split('_')[0].lower())
            # im2 = array(im)
            # imshow(im2) 

            # row, col = im2.shape[0:2]
            # print row, col   
            # data[ = im.getdata()
            # data = np.matrix(data)

            # label = file.split('_')
            # label = label[0].lower()
            # ALL_label.append(label)
            # ALL_data.append(data)
#            print ALL_data
#            print label
#            data.show()
#        print ALL_data
        data /= 255
        return data, label

    def preprocess(self):
        pass
        
    def get_ocr_net(self, x, y):

        print ('start CNN model...')
        model = Graph()
        #input       1通道 150*40图片
        model.add_input(name='input', input_shape=(1, 150, 40))
        #conv1
        model.add_node(Convolution2D(64, 5, 5), name='conv1', input='input')
        model.add_node(MaxPooling2D(pool_size=(2, 2)), name='pool1', input='conv1')
        #conv2
        model.add_node(Convolution2D(64, 3, 3), name='conv2', input='pool1')
        model.add_node(MaxPooling2D(pool_size=(2,2)), name='pool2', input='conv2')
        model.add_node(Dropout(0.25), name='drop', input='pool2')
        #
        model.add_node(Flatten(), name='flatten', input='drop')
        #conv * 4   分别对应验证码四个字符
        model.add_node(Dense(512), name='D', input='flatten')

        model.add_node(Dense(self.nb_class), name='C1', input='D')
        model.add_node(Dense(self.nb_class), name='C2', input='D')
        model.add_node(Dense(self.nb_class), name='C3', input='D')
        model.add_node(Dense(self.nb_class), name='C4', input='D')

        model.add_output(name='output', inputs=['C1', 'C2', 'C3', 'C4'])
        # #dense1
        # model.add_node(Dense(256), name='D1', input='flatten')
        # model.add_node(Dropout(0.5), name='drop', input='D1')
        # #dense2
        # model.add_node(Dense(10), name='result', input='drop')
        # model.add_node(Activation('softmax'))
        #output
        model.add_output(name='out', input='result')

        sgd = SGD(lr=0.01, decay=1e-6,
                momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy',
                optimizer=sgd, class_mode="categorical")

        early_Stop = EarlyStopping(monitor='acc',#'val_loss', 
                                        verbose=2, mode='auto', patience=10)
        batch_size = 32
        train_num = int(len(x) * 0.8)
        model.fit(x[:train_num], y[:train_num], 
                   batch_size = batch_size, 
                    shuffle=True, 
                    verbose = 2,
                    #validation_split=0.1,
                    validation_data=[x[train_num:], y[train_num:]],
                    #callbacks=[early_Stop],
                  nb_epoch=10)
        print ('CNN model training is ok!!!')
        #测试模型
        score, acc = model.evaluate(x[train_num:], y[train_num:], 
                                    batch_size = batch_size)
        print('Test accuracy:', acc)
        return model       

#=============================
# jhy = gen_rand()
# jhy_num = get_label(jhy)
# captcha = ImageCaptcha(fonts=['./data/Xerox.ttf'])
#cap = gen_sample(captcha, 30, 30)
if __name__ == '__main__':
    DL = DL_FOR_Captcha()
    #model = DL.get_ocr_net()
    x, y= DL.load_data()
    xx= x[1]

