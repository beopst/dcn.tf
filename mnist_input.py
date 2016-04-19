from __future__ import division

import gflags
import os, glob
import time, random
import tensorflow as tf
import numpy as np
from skimage import io
from skimage import img_as_float

import ipdb
FLAGS = gflags.FLAGS

def load_data(src,shuffle=True):
    """
        load lines in csv file as a list of strings
    """

    imgs = [img for img in glob.glob(os.path.join(src,'*.png'))]

    x = np.zeros((len(imgs),100,100), dtype=np.float32)
    y = np.zeros(len(imgs), dtype=np.int64)

    for idx, img in enumerate(imgs):
        im = io.imread(img,1)
        im = img_as_float(im) - 0.5 # rescale from [0,255] to [-0.5,0.5]

        label = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        
        x[idx] = im
        y[idx] = label

    x = np.expand_dims(x,3)
    data = zip(x,y)

    if shuffle: random.shuffle(data)

    return data



