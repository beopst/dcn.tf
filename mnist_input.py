from __future__ import division

import os, glob
import random
import numpy as np
from skimage import io
from skimage import img_as_float

def load_data(src,shuffle=True):
    """ Load data from directories.
    """

    imgs = [img for img in glob.glob(os.path.join(src,'*.png'))]

    x = np.zeros((len(imgs),100,100), dtype=np.float32)
    y = np.zeros(len(imgs), dtype=np.int64)

    for idx, img in enumerate(imgs):
        im = io.imread(img,1)
        im = img_as_float(im) # rescale from [0,255] to [0,1]

        label = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        
        x[idx] = im
        y[idx] = label

    x = np.expand_dims(x,3)
    data = zip(x,y)

    if shuffle: random.shuffle(data)

    return data



