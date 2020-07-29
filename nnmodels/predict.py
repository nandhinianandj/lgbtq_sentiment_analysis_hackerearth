# -*- coding: utf-8 -*-
# -.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.#

#* File Name : predict.py
#
#* Purpose :
#
#* Creation Date : 23-10-2018
#
#* Last Modified : Tue 06 Nov 2018 01:18:23 AM EST
#
#* Created By :

#_._._._._._._._._._._._._._._._._._._._._.#
import argparse
import json
import os
import matplotlib
import numpy as np
import subprocess


from keras.models import load_model
from PIL import Image
from pprint import pprint

import settings

def main(imgfile):
    modelf = './models/final_model.h5'
    f,fname,ext = modelf.split('.')
    conff = '.' + fname + '_params.json'
    model = load_model(modelf)
    with open(conff, 'r') as fd:
        confs = json.load(fd)
    pprint("Model was trained with the following configuration")
    pprint(confs)
    if 'gray' in confs['filename']:
        im_open_mode = 'L'
    else:
        im_open_mode = 'RGB'

    with Image.open(imgfile).convert(im_open_mode) as img:
        img_arr =  np.array(img.getdata()).reshape((1,400,300,1))
    #pprint(confs['class_maps'])

    predictions = model.predict(img_arr)[0]
    pprint("Model predicted probabilities")
    pprint({'crystalline': predictions[2],
            'porous': predictions[0],
            'fibrous': predictions[1]}
            )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read the annotations and guess the size of kidney stone')
    parser.add_argument('--image_file',  type=str.lower,
                        help='Full path of the image to predict texture of',
                        default='data/standard_images/porous/porous_0131.jpg')
    opt = parser.parse_args()
    imgfile = opt.image_file
    main(imgfile)
