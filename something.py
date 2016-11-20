# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:53:25 2016

@author: Jhy1993
"""
image_model = Sequential()
image_model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(3, 100, 100)))
image_model.add(Activation('relu'))
image_model.add(Convolution2D(32, 3, 3))
image_model.add(Activation('relu'))
image_model.add(MaxPooling2D(pool_size=(2, 2)))
image_model.add(Flatten())
image_model.add(Dense(128))

language_model = Sequential()
language_model.add(Embedding(vocab_size, 256, input_length=max_caption_len))
language_model.add(GRU(output_dim=128, return_sequences=True))
language_model.add(TimeDistributed(128))

image_model.add(RepeatVector(max_caption_len))

model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=-1))
model.add(GRU(256, return_sequences=False))

model.add(Dense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.fit([images, partial_captions], next_words, batch_size=16, nb_epoch=100,
            validation_data=([x1_val, x2_val], y_val))
#====================多输入 多输出=================
main_input = Input(shape=(100, ), dtype='int32', name='main_input')
aux_input = Input(shape=(5, ), name='aux_input')





model = Model(input=[main_input, aux_input],
                output=[main_loss, aux_loss])
ES = EarlyStopping(monitor='val_loss', patience=2)
model.compile(optimizer='rmsprop',
            loss={'main_loss': 'binary_crossentropy', 'aux_loss': 'binary_crossentropy'})
hist = model.fit({'main_input': main_data, 'aux_input': aux_data},
        {'main_output': main_label, 'aux_output': aux_label},
        nb_epoch=50, batch_size=32,
        validation_split=0.2,
        callbacks=[ES])
print(hist.history)


#===============多GPU================================
with tf.device('/gpu:0'):
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = LSTM(32)(x)

with tf.device('/gpu:1'):
    x = tf.placeholder(tf.float32, shape=(None, 20, 64))
    y = LSTM(32)(x)

#====================分布式=======================

#=================generator===================
def generate_from_file(path):
    while 1:
        f = open(path)
        for line in f:
            x, y = process_line(line)
            yield x, y
        f.close()
model.fit_generator(generate_from_file('11.txt'),
                        samples_per_epoch=1000,
                        nb_epoch=10)
model.evaluate_generator(generate_from_file('2.txt'),
                        val_samples=1000)
#======================Graph model===============
a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(input=a, output=b)
model = Model(input=[a1, a2], output=[b1, b2])
model.compile(optimizer='rmsprop',
            loss='binary_crossentropy',
            loss_weights=None,
            sample_weight_mode=None)
model.fit({'a1': data_a1, 'a2': data_a2},
            {'b1': data_b1, 'b2': data_b2},
            nb_epoch=10,
            verbose=1,
            validation_data=None,
            class_weight=None,
            sample_weight=None)
model.fit_generator(generator,
                    samples_per_epoch=1000,
                    nb_epoch=10,
                    verbose=1,
                    validation_data=None,
                    nb_val_samples=None,
                    class_weight={},
                    )

#=====================================
from keras.utils.layer_utils import layer_from_config
config = layer.get_config()
layer = layer_from_config(config)
#=====================================
