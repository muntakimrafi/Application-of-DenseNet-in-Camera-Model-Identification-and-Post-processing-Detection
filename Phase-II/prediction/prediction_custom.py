#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 02:54:56 2017

@author: root
"""
import tensorflow as tf
from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D, concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as Kyyfffffffff
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from dense_all_keras import DenseNetImageNet201
from sklearn.metrics import log_loss
#from scale_layers import Scale
import keras
from keras.callbacks import LearningRateScheduler
from skimage.restoration import denoise_wavelet
from skimage import img_as_float,img_as_uint
from keras.utils import np_utils
import numpy as np
from scipy.misc import imread
#from imageio import imread
import math
from scipy import signal
from statistics import mode
import random
from dense_all_keras import DenseNetImageNet201
from keras.applications.densenet import DenseNet201

img_rows = 256
img_cols = 256
num_classes = 27
channel = 3
batch_size=16

base_model = DenseNetImageNet201(input_shape = None,
                        bottleneck=True,
                        reduction=0.5,
                        dropout_rate=0.0,
                        weight_decay=1e-4,
                        include_top=False,
                        weights=None,
                        input_tensor=None,
                        pooling=None,
                        classes=None
                        )
x = base_model.output
x= GlobalAveragePooling2D()(x)

zz = Dense(10, activation = 'softmax')(x)


final_model = Model(inputs = base_model.input, outputs = zz)

final_model.load_weights('')


def patch_creator(img):
    
    p=0;
    img_in = imread(img)

    m, n = img_in.shape[0:2]                                                                                                                                                  
    a, b = 512//img_rows, 512//img_cols
    all_patch = np.zeros((a*b, img_rows, img_cols, 3))
    for k in range(a):
        for l in range (b):
            all_patch[p,:,:,:] = img_in[(k*img_rows):(k+1)*img_rows, (l*img_cols):(l+1)*img_cols, :]
            p+=1
            
    return all_patch

y_gen2 = np.zeros((len(test_imdir),1))
y_pred_2 = np.zeros((len(test_imdir),batch_size,10))
for i in range(len(test_imdir)):
    
    patch = patch_creator(test_imdir[i])  
    y_pred_2[i] = final_model.predict(patch, batch_size = batch_size)

    print(i)
y_pred2 = np.average(y_pred_2,axis = 1)
y_gen2 = y_pred2.argmax(axis=1)+1
 
