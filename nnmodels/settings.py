import pathlib

import numpy as np
import os

LABELS = ("kidney_stone",)
MODELS_BASE_PATH = os.environ.get('MODELS_BASE_PATH', './data/models')
MODEL_PERF_TRACKER = os.environ.get(
    'MODEL_PERF_TRACKER', './data/models/models_performance.csv')
MODEL_PERF_FIELDS = ['timestamp', 'modelname', 'modelfile',
                     'validation_loss', 'validation_accuracy',
                     'batch_size', 'epochs']

import os

INPUT_DIR='./data/texture_samples'
MODELS_BASE_PATH='./models'
PLOTS_DIR='./plots'
STANDARD_IMAGES_DIR='./data/standard_images'

CLASSES = ['porous', 'fibrous', 'crystalline']#, 'blotchy']
NUM_CLASSES = len(CLASSES)

IMAGE_SIZE=(400, 300)
MODEL_PERF_TRACKER='models_comparison.csv'
TEST_DATASET_FRAC=0.2
## Model types
POSSIBLE_MODELS = ['keras-cnn', 'keras-inception']

## Input tyes
POSSIBLE_INPUTS=['color-raw-images', 'gray-raw-images',
                 'color-gabor-filtered', 'gray-gabor-filtered',
                 'gray-wavelet-coeffs']

## Optimizer types
POSSIBLE_OPTIMIZERS=['adagrad', 'adadelta', 'sgd', 'rmsprop', 'adam']

## NN model settings
BATCH_SIZE=18
EPOCHS=100
AUGMENTED_SAMPLE_SZ = 5000
# FEATURE filter settings
GABOR_KSIZE=39
WAVLET='bior2.2'
WAVLET_LVL=None  #Use None for default max values
