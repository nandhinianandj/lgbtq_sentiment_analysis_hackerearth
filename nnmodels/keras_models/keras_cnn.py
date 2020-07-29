# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : keras_cnn.py
#
#* Purpose : Just a wrapper for a CNN network based on keras and some dynamic learning rate utils gathered from various blogs
#
#* Creation Date : 11-10-2018
#
#* Last Modified : Wed 07 Nov 2018 04:05:50 PM IST
#
#* Created By :
# Some sources: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
#_._._._._._._._._._._._._._._._._._._._._.#

import math
import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras
import numpy as np

from keras.initializers import Constant
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from keras.layers.merge import concatenate
from keras.models import Sequential, Model

import keras_utils as ku
import settings

def get_keras_CNN(input_shape, num_classes, initial_conv_filt_sz=128,
                  cnn_layers=2, initial_conv_kern_sz=11, conv_activ_func='relu',
                  optim_type='sgd', reduce_conv_filt_sz=True):
    model = Sequential()
    model.add(Conv2D(initial_conv_filt_sz,
        	     kernel_size=(initial_conv_kern_sz, initial_conv_kern_sz), strides=(1, 1),
                     activation=conv_activ_func, #'relu',
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    for i in range(cnn_layers):
        new_layer_sz = int(initial_conv_filt_sz /2**(i+1)) if reduce_conv_filt_sz else initial_conv_filt_sz
        new_kern_sz = max(1, int(initial_conv_kern_sz - (i+1)*1))
        # n layers of normal convolution
        model.add(Conv2D(new_layer_sz, (3,3), activation=conv_activ_func))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        print("Added layer no:",model.output)

    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

    return ku.compile_model(model, optim_type)

