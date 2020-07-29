# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : keras_utils.py
#
#* Purpose :
#
#* Creation Date : 07-11-2018
#
#* Last Modified : Fri 30 Nov 2018 04:17:10 AM EST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#
import keras
import math
import matplotlib
import os

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from keras.utils import plot_model

import settings
def compile_model(model, optim_type, loss_type):
    if optim_type=='sgd':
        optimizer=keras.optimizers.SGD(lr=0.0005, momentum=0.99, )
    if optim_type=='adagrad':
        optimizer=keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    if optim_type=='adadelta':
        optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    if optim_type=='rmsprop':
        optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    if optim_type=='adam':
        optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    if loss_type=='sparse_categorical_crossentropy':
        loss_func=keras.losses.sparse_categorical_crossentropy
    if loss_type=='categorical_crossentropy':
        loss_func=keras.losses.categorical_crossentropy
    if loss_type=='binary_crossentropy':
        loss_func=keras.losses.binary_crossentropy
    if loss_type=='poisson':
        loss_func=keras.losses.poisson
    if loss_type=='kullback_leibler_divergence':
        loss_func=keras.losses.kullback_leibler_divergence
    if loss_type=='cosine_proximity':
        loss_func=keras.losses.cosine_proximity

    model.compile(loss=loss_func,
                  optimizer=optimizer, metrics=['categorical_accuracy'])
    return model


def eval_keras_model(model, x_test, y_test, model_params, history=None):

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score

def plot_training_histories(model_params, acc_history=None, loss_history=None):
    if not model_params:
        raise ValueError("model_params required dictionary")
    if not any([acc_history, loss_history,]):
        raise ValueError("One of acc_history, loss_history is necessary")
    for history in [acc_history, loss_history]:
        if history:
            label = str(history)
            plt.plot(range(1,model_params['epochs']+1), history.val)
            plt.xlabel('Epochs')
            plt.ylabel(label)
            plt.legend()
            imgfile = os.path.split(model_params['modelfile'])[-1].split('.')[0] + '.jpg'
            plt.savefig(os.path.join(settings.PLOTS_DIR, '{}'.format(imgfile)))

def plot_keras_model(model, model_params):
    imgfile = os.path.split(model_params['modelfile'])[-1].split('.')[0] + '_arch.jpg'
    plot_model(model, to_file=os.path.join(settings.PLOTS_DIR, imgfile))

def exp_decay(epoch):
   initial_lrate = 0.1
   k = 0.1
   lrate = initial_lrate * exp(-k*t)
   return lrate

def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate

class LossHistory(keras.callbacks.Callback):
    def __str__(self):
        return "Training Loss"
    def __repr__(self):
        return "Training Loss"
    def on_train_begin(self, logs={}):
       self.val = []
       self.lr = []

    def on_epoch_end(self, batch, logs={}):
       self.val.append(logs.get('loss'))
       self.lr.append(step_decay(len(self.val)))


class AccuracyHistory(keras.callbacks.ModelCheckpoint):
    def __init__(self, file_path, **kwargs):
        super(AccuracyHistory, self).__init__(filepath=file_path, **kwargs)

    def __str__(self):
        return "Training Accuracy"
    def __repr__(self):
        return "Training Accuracy"
    def on_train_begin(self, logs={}):
        self.val = []

    def on_epoch_end(self, batch, logs={}):
        self.val.append(logs.get('acc'))
