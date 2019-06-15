#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 23:58:15 2017

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
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import log_loss
#from scale_layers import Scale
import keras
from keras.callbacks import LearningRateScheduler
from skimage.restoration import denoise_wavelet
from skimage import img_as_float,img_as_uint
from keras.utils import np_utils
import numpy as np
#from scipy.misc import imread
from imageio import imread
import math
from scipy import signal
import random
from dense_all_keras import DenseNetImageNet201
from densenet_201_new import get_model

#
#img_cols1 = 256

img_rows = 256
img_cols = 256

#img_rows2 = 128
#img_cols2 = 128

def get_all_patch(img, size):
    num = int(256//size)
    all_patch = np.zeros((num*num, size, size, 3))
    p=0
    for i in range(num):
        for j in range(num):
            all_patch[p]=img[i+size,j+size,:]
            p+=1
    return all_patch

combined_train = list(zip(train_imdir, train_label))
random.shuffle(combined_train)

train_imdir[:], train_label[:] = zip(*combined_train)

combined_val= list(zip(val_imdir, val_label))
random.shuffle(combined_val)

val_imdir[:], val_label[:] = zip(*combined_val)

def augment(src, choice):
            
    if choice == 0:
        src = np.rot90(src, 1)
                
    if choice == 1:
        src = src
                
    if choice == 2:
        src = np.rot90(src, 2)
                
    if choice == 3:
        src = np.rot90(src, 3)
    return src

from tqdm import tqdm

def generate_processed_batch(inp_data,label,batch_size = 50):

    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_label = np.zeros((batch_size, 10))
    while 1:
        for i_data in range(0,len(inp_data),batch_size):
            for i_batch in range(batch_size):
                print(i_data+i_batch)
                img = imread(inp_data[i_data+i_batch])
                img = augment(img, np.random.randint(4))
                
                a = 256 - img_rows
                d = np.random.randint(0,a)
                img = img[d:d+img_rows,d:d+img_rows,:]
                
                lab = np_utils.to_categorical(label[i_data+i_batch],10)

                batch_images[i_batch] = img
                batch_label[i_batch] = lab

            yield batch_images, batch_label



batchsize = 16

    

model_64_path = '' #model trained on 256X256 patches
model_64 = get_model(model_64_path)
model_128_path = '' #same model
model_128 = get_model(model_128_path)
model_256_path = '' #same model
model_256 = purana_model(model_256_path)

model_256_feat = np.zeros((200000,1920))
model_128_feat = np.zeros((200000,1920))
model_64_feat = np.zeros((200000,1920))

print('start')

for i in tqdm(range(200000)):
    img = imread(train_imdir[i])
    img_128 = get_all_patch(img,128)
    img_64 = get_all_patch(img, 64)
    img_256 = np.expand_dims(img, axis= 0)
    model_256_feat[i] = model_256.predict(img_256)
    model_128_feat[i] = np.average(model_128.predict_on_batch(img_128), axis = 0)
    model_64_feat[i] = np.average(model_64.predict_on_batch(img_64), axis = 0)

np.save('model_128_feat', model_128_feat)
np.save('model_64_feat', model_64_feat)
np.save('model_256_feat', model_256_feat)
np.save('l2_model_label', train_label)
    
