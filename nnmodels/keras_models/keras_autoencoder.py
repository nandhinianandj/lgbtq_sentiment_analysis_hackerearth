# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : keras_autoencoder.py
#
#* Purpose :
#
#* Creation Date : 17-10-2018
#
#* Last Modified : Wed 17 Oct 2018 04:57:23 PM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#


from keras.layers import Input, Dense
from keras.models import Model

def get_autoencoder(input_shape, encoding_dim):
    # this is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=input_shape)
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(784, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder


