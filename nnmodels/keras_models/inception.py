# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name :
#
#* Purpose :
#
#* Creation Date : 18-10-2018
#
#* Last Modified : Thu 18 Oct 2018 04:02:12 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten

import cv2
import numpy as np
from keras import backend as K
from keras.utils import np_utils

import math
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler


def inception_module(model,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):

    # conv 1x1
    model.add(Conv2D(filters_1x1, (1, 1), padding='same', activation='relu',
        kernel_initializer=kernel_init, bias_initializer=bias_init))

    #  conv_3x3
    model.add(Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu',
        kernel_initializer=kernel_init, bias_initializer=bias_init))
    #conv_3x3
    model.add(Conv2D(filters_3x3, (3, 3), padding='same', activation='relu',
        kernel_initializer=kernel_init, bias_initializer=bias_init))

    # conv_5x5
    model.add(Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu',
        kernel_initializer=kernel_init, bias_initializer=bias_init))
    #conv_5x5
    model.add(Conv2D(filters_5x5, (5, 5), padding='same', activation='relu',
        kernel_initializer=kernel_init, bias_initializer=bias_init))

    # pool_proj
    model.add(MaxPool2D((3, 3), strides=(1, 1), padding='same'))
    # pool_proj
    model.add(Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu',
        kernel_initializer=kernel_init, bias_initializer=bias_init))

    return model

def get_inception_model(input_shape):
    input_layer = Input(shape=input_shape)

    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)

    model = Sequential()
    model.add(Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu',
        name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init))
    model.add(MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_1_3x3/2'))
    model.add(Conv2D(64, (1, 1), padding='same', strides=(1, 1), activation='relu',
                name='conv_2a_3x3/1'))
    model.add(Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu',
                name='conv_2b_3x3/1'))
    model.add(MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_2_3x3/2'))
    model.add(inception_module(model,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_3a'))

    model.add(inception_module(model,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=192,
                         filters_5x5_reduce=32,
                         filters_5x5=96,
                         filters_pool_proj=64,
                         name='inception_3b'))

    model.add(MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2'))

    model.add(inception_module(model,
                         filters_1x1=192,
                         filters_3x3_reduce=96,
                         filters_3x3=208,
                         filters_5x5_reduce=16,
                         filters_5x5=48,
                         filters_pool_proj=64,
                         name='inception_4a'))
    model.add(AveragePooling2D((5, 5), strides=3))
    model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(10, activation='softmax', name='auxilliary_output_1'))

    model.add(inception_module(model,
                         filters_1x1=160,
                         filters_3x3_reduce=112,
                         filters_3x3=224,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4b'))

    model.add(inception_module(model,
                         filters_1x1=128,
                         filters_3x3_reduce=128,
                         filters_3x3=256,
                         filters_5x5_reduce=24,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4c'))

    model.add(inception_module(model,
                         filters_1x1=112,
                         filters_3x3_reduce=144,
                         filters_3x3=288,
                         filters_5x5_reduce=32,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_4d'))


    model.add(AveragePooling2D((5, 5), strides=3))
    model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(10, activation='softmax', name='auxilliary_output_2'))

    model.add(inception_module(model,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_4e'))

    model.add(MaxPool2D((3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2'))

    model.add(inception_module(model,
                         filters_1x1=256,
                         filters_3x3_reduce=160,
                         filters_3x3=320,
                         filters_5x5_reduce=32,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5a'))

    model.add(inception_module(model,
                         filters_1x1=384,
                         filters_3x3_reduce=192,
                         filters_3x3=384,
                         filters_5x5_reduce=48,
                         filters_5x5=128,
                         filters_pool_proj=128,
                         name='inception_5b'))

    model.add(GlobalAveragePooling2D(name='avg_pool_5_3x3/1'))

    model.add(Dropout(0.4))

    model.add(Dense(10, activation='softmax', name='output'))
    return model
