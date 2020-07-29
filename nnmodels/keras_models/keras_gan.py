# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : keras_gan.py
#
#* Purpose :
#
#* Creation Date : 16-10-2018
#
#* Last Modified : Wed 17 Oct 2018 01:06:25 AM IST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#

# Code taken from here https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0

def discriminator(input_shape):
    D = Sequential()
    depth = 64
    dropout = 0.4
    D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
                padding='same', activation=LeakyReLU(alpha=0.2)))
    D.add(Dropout(dropout))
    D.add(Conv2D(depth*2, 5, strides=2, padding='same',\
    activation=LeakyReLU(alpha=0.2)))
    D.add(Dropout(dropout))
    D.add(Conv2D(depth*4, 5, strides=2, padding='same',\
    activation=LeakyReLU(alpha=0.2)))
    D.add(Dropout(dropout))
    D.add(Conv2D(depth*8, 5, strides=1, padding='same',\
    activation=LeakyReLU(alpha=0.2)))
    D.add(Dropout(dropout))
    # Out: 1-dim probability
    D.add(Flatten())
    D.add(Dense(1))
    D.add(Activation('sigmoid'))
    D.summary()
    return D

def generator( input_shape):
    G = Sequential()
    dropout = 0.4
    depth = 64+64+64+64
    dim = 7
    # In: 100
    # Out: dim x dim x depth
    G.add(Dense(dim*dim*depth, input_dim=100))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Reshape((dim, dim, depth)))
    G.add(Dropout(dropout))
    # In: dim x dim x depth
    # Out: 2*dim x 2*dim x depth/2
    G.add(UpSampling2D())
    G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(UpSampling2D())
    G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
    G.add(Conv2DTranspose(1, 5, padding='same'))
    G.add(Activation('sigmoid'))
    G.summary()
    return G

def get_gan(input_shape):
    optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
    model = Sequential()
    model.add(generator(input_shape))
    model.add(discriminator(input_shape))
    model.compile(loss='binary_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
    return model
