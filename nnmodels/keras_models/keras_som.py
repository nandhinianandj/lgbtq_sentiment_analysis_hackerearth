# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : keras_som.py
#
#* Purpose :
#
#* Creation Date : 16-10-2018
#
#* Last Modified : Tue 16 Oct 2018 11:14:56 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras
import numpy as np

from keras.initializers import Constant
from keras.layers import Reshape, Activation, Conv2D, Conv2DTranspose, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.models import Sequential, Model

import settings

def get_keras_som(input_shape):
    model = Sequential()

    pass
