import argparse
import csv
import datetime
import keras
import numpy as np
import os
import tensorflow as tf

from keras.callbacks import LearningRateScheduler

import settings
import read_data as rd
from toolbelt import modelStorageUtils as utils
import keras_utils as ku
import keras_models as km
#TODO: See if you can find a visualization method that's agnosit of underlying library, just take a
# model and draw an architecture diagram somehow.

###################################
# TensorFlow wizardry for handling OOM errors
#config = tf.ConfigProto()
#
## Don't pre-allocate memory; allocate as-needed
#config.gpu_options.allow_growth = True
#
## Only allow a total of half the GPU memory to be allocated
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#
## Create a session with the above options specified.
#import keras.backend.tensorflow_backend as K
#K.set_session(tf.Session(config=config))
###################################


def main(model_type, input_type, optim_type, batch_size, epochs, augment=False):
    # Retrieve input adat and create train test split
    model_params = dict()

    if not os.path.exists(settings.MODEL_PERF_TRACKER):
        with open(settings.MODEL_PERF_TRACKER, 'w') as csvfile:
            fieldnames = ['timestamp', 'modelname', 'modelfile', 'validation_loss',
            'validation_accuracy', 'batch_size', 'epochs']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    train_datagen, test_datagen, class_maps, data_shape = rd.read_inputs(input_type, augment, batch_size)

    if 'gabor-filtered' in input_type:
        input_type = '%s-%d-kernel'%(input_type, settings.GABOR_KSIZE)

    modelname = '%s_%s_%s-textures'%(model_type, input_type, settings.NUM_CLASSES)
    if model_type == 'keras-cnn':
        input_shape = data_shape
        loss_history = ku.LossHistory()
        acc_history = ku.AccuracyHistory(file_path=os.path.join(settings.MODELS_BASE_PATH, 'tmp'))
        lrate = LearningRateScheduler(ku.step_decay)
        callbacks_list = [loss_history, acc_history, lrate]
        model = km.keras_cnn.get_keras_CNN(input_shape, settings.NUM_CLASSES,
				initial_conv_filt_sz=256,
                                initial_conv_kern_sz=5,
                                conv_activ_func='relu',
				cnn_layers=4, reduce_conv_filt_sz=False)
        model.fit_generator(train_datagen,
                      epochs=epochs,
                      verbose=1,
                      validation_data=test_datagen,
                      callbacks=callbacks_list)
        model_params.update({
                          'model_type': 'keras_conv_2d',
                          'output_type': 'Texture class encoded by %s'%('-'.join(settings.CLASSES)),
                          'input_metadata': 'image wavelet transformed array of size XxYx3',
                          'model_confs': model.get_config(),
                          'class_maps': class_maps,
                          'modelname': modelname,
                          'batch_size': batch_size,
                          'epochs': epochs
                            })
        val_data, test_targets = test_datagen.__getitem__(2)
        score = ku.eval_keras_model(model, val_data, test_targets ,history=acc_history, model_params=model_params)
        model_params['score'] = score
        modelfile = utils.dump_model(model, '%s_%s'%(model_type, input_type), model_params)
        model_params.update({'modelfile': modelfile})
        ku.plot_training_histories(model_params, acc_history=acc_history, loss_history=loss_history)

    if model_type == 'keras-inception':
        loss_history = ku.LossHistory()
        acc_history = ku.AccuracyHistory(file_path=os.path.join(settings.MODELS_BASE_PATH, 'tmp'))
        lrate = LearningRateScheduler(ku.step_decay)
        callbacks_list = [loss_history, acc_history]#, lrate]
        model = km.get_inception_model(data_shape, optim_type)

        #model.fit_generator(train_datagen,
        #                        epochs=epochs, verbose=1,
        #                        validation_data=test_datagen,
        #                        callbacks=callbacks_list)
        train_x, train_y = train_datagen.__getitem__(int(settings.AUGMENTED_SAMPLE_SZ/batch_size))
        test_x, test_y = test_datagen.__getitem__(int(settings.AUGMENTED_SAMPLE_SZ*\
                                                    settings.TEST_DATASET_FRAC/batch_size))

        # Reverse to_categorical
        #targets = np.array([ np.argmax(each) for each in train_y])
        #ttargets = np.array([np.argmax(each) for each in test_y])

        targets = train_y #[train_y[:,0], train_y[:,1], train_y[:,2]]
        ttargets = test_y #[test_y[:,0], test_y[:,1], test_y[:,2]]

        model.fit(train_x, targets, batch_size=batch_size,
                  epochs=epochs, verbose=1,
                  validation_data=(test_x, ttargets),
                  callbacks=callbacks_list)

        model_params.update({
                          'model_type': 'keras_inception',
                          'output_type': 'Texture class encoded by %s'%('-'.join(settings.CLASSES)),
                          'input_metadata': 'image of size XxYx3',
                          'model_confs': model.get_config(),
                          'class_maps': class_maps,
                          'modelname': modelname,
                          'epochs': epochs
                            })
        score = ku.eval_keras_model(model, test_x, ttargets ,history=acc_history, model_params=model_params)
        model_params['score'] = score
        modelfile = utils.dump_model(model, '%s_%s'%(model_type, input_type), model_params)
        model_params.update({'modelfile': modelfile})
        ku.plot_keras_model(model, model_params)
        ku.plot_training_histories(model_params, acc_history=acc_history, loss_history=loss_history)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read images from the settings.STANDARD_IMAGES_DIR,\
                        and train a CNN model with keras to classify the textures')

    parser.add_argument('--model_type', choices=settings.POSSIBLE_MODELS, type=str.lower,
                        help='Types of model to train', default='keras-cnn')
    parser.add_argument('--input_type', choices=settings.POSSIBLE_INPUTS, type=str,
                        help='Type of pre-processed inputs to pass to the NN for training',
                        default='gray-wavelet-coeffs')
    parser.add_argument('--augment_images', dest='augment', action='store_true',
                        help='Augment images or not')
    parser.add_argument('--optim_type', choices=settings.POSSIBLE_OPTIMIZERS, type=str.lower,
                        help='What kind of optimizer to use to train the model', default='sgd')

    parser.add_argument('--batch_size', type=int, help='Size of batches for training',
                        default=settings.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for',
                        default=settings.EPOCHS)
    opt = parser.parse_args()

    model_type = opt.model_type
    input_type = opt.input_type
    batch_size = opt.batch_size
    optim_type = opt.optim_type
    epochs = opt.epochs
    augment = opt.augment

    main(model_type, input_type, optim_type, batch_size, epochs, augment)
