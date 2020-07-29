import Augmentor
import keras
import numpy as np
import os
import shutil
import subprocess

from sklearn.model_selection import train_test_split
from PIL import Image

import settings

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    # Taken from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self, list_IDs, labels, batch_size=32, dim=settings.IMAGE_SIZE, n_channels=1,
                 n_classes=settings.NUM_CLASSES, im_open_mode='L', shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.labels = dict(zip(self.list_IDs, labels))
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.im_open_mode = im_open_mode
        self.data_shape = (*self.dim, self.n_channels)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __iter__(self):
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            with Image.open(ID).convert(self.im_open_mode) as img:
                X[i,] = np.array(img.getdata()).reshape(self.data_shape)

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def image_generators(input_type, channels, im_open_mode, batch_size, augment=False):

    if augment:
        # Create a image augmentation pipeline
        # Packing in as many augmentations as possible without checking results for
        # semantic/meaningfulness
        pipeline = Augmentor.Pipeline(settings.STANDARD_IMAGES_DIR)
        pipeline.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
        pipeline.rotate(probability=0.3, max_left_rotation=5, max_right_rotation=5)
        pipeline.sample(settings.AUGMENTED_SAMPLE_SZ)
        source_dir = os.path.join(settings.STANDARD_IMAGES_DIR, 'output')
    else:
        source_dir = settings.STANDARD_IMAGES_DIR

    # Read the images from the fs.. standard or augmented
    class_maps = dict()
    files = list()
    labels = list()
    classes = settings.CLASSES
    for cls_num, cls in enumerate(classes):
        class_maps[cls_num] = cls
        for fil in os.listdir(os.path.join(source_dir, cls)):
            fnam = os.path.join(source_dir, cls, fil)
            files.append(fnam)
            labels.append(cls_num)

    # Split into train and test dataset
    x_train, x_test, y_train, y_test = train_test_split(files, labels, test_size=settings.TEST_DATASET_FRAC)

    # Create keras compatible generators
    if input_type in ['color-raw-images', 'gray-raw-images']:
        train_datagen = DataGenerator(x_train, y_train, n_channels=channels,
                                        batch_size=batch_size,
                                        im_open_mode=im_open_mode)
        test_datagen = DataGenerator(x_test, y_test, n_channels=channels,
                                        batch_size=batch_size,
                                        im_open_mode=im_open_mode)
    elif 'gabor-filtered' in input_type:
        train_datagen = GaborFilteredDataGenerator(x_train, y_train,
                                                   dim=settings.IMAGE_SIZE,
                                                   n_channels=channels,
                                                   batch_size=batch_size,
                                                   im_open_mode=im_open_mode)
        test_datagen = GaborFilteredDataGenerator(x_test, y_test,
                                                  dim=settings.IMAGE_SIZE,
                                                  n_channels=channels,
                                                  batch_size=batch_size,
                                                  im_open_mode=im_open_mode)
    elif 'wavelet-coeffs' in input_type:
        train_datagen = WaveletCoeffsDataGenerator(x_train, y_train,
                                                   dim=settings.IMAGE_SIZE,
                                                   n_channels=channels,
                                                   batch_size=batch_size,
                                                   im_open_mode=im_open_mode)
        test_datagen = WaveletCoeffsDataGenerator(x_test, y_test,
                                                  dim=settings.IMAGE_SIZE,
                                                  n_channels=channels,
                                                  batch_size=batch_size,
                                                  im_open_mode=im_open_mode)
    else:
        raise ValueError("invalid input_type passed")
    return (train_datagen, test_datagen, class_maps)

def read_inputs(input_type, augment=None, batch_size=None):
    """Construct augmented image inputs for Texture classifier training.

    Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1/3] size.
    labels: Labels. 1D tensor of [batch_size] size.

    #TODO: This function is ugly from a pure-functions functional programming view point, clean it
    up.
    """
    # Convert images to required color format and required size
    standardize(input_type )

    # Create 4-D empty arrays to append images(w x h x 3 x num_images)
    train_imgs, test_imgs = list(), list()
    train_lbls, test_lbls = list(), list()

    if input_type.startswith('color'):
        im_open_mode = 'RGB'
        channels = 3
    else:
        im_open_mode = 'L'
        channels = 1
    data_shape = (settings.IMAGE_SIZE[0], settings.IMAGE_SIZE[1], channels)

    train_datagen, test_datagen, class_maps = image_generators(input_type, channels, im_open_mode, batch_size, augment)
    return train_datagen, test_datagen, class_maps, data_shape

def standardize(input_type, input_dir=settings.INPUT_DIR,
                standard_dir=settings.STANDARD_IMAGES_DIR):
    classes = settings.CLASSES
    shutil.rmtree(standard_dir)
    os.mkdir(standard_dir)
    for cls in classes:
        if not os.path.exists(os.path.join(standard_dir, cls)):
            os.mkdir(os.path.join(standard_dir, cls))
        else:
            continue
        for fil in os.listdir(os.path.join(input_dir, cls)):
            fnam = os.path.join(input_dir, cls, fil)
            out_fnam = os.path.join(standard_dir, cls, fil)
            resize_image(input_type, fnam, out_fnam)

def resize_image(input_type, input_fnam, out_fnam):
    width, height = settings.IMAGE_SIZE
    if 'gray' in input_type:
        convert_cmd = ['convert', '%s'%input_fnam,
                   '-colorspace Gray',
                       '-resize', '{}x{}\>'.format(width, height),
                       '-resize', '{}x{}\<'.format(width, height),
                       '-extent', '{}x{}'.format(width, height),
                       '%s'%out_fnam
                      ]
    else:
        convert_cmd = ['convert', '%s'%input_fnam,
                       '-resize', '{}x{}\>'.format(width, height),
                       '-resize', '{}x{}\<'.format(width, height),
                       '-extent', '{}x{}'.format(width, height),
                       '%s'%out_fnam
                      ]
    subprocess.run(' '.join(convert_cmd), shell=True)
