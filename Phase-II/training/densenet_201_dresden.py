#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 23:58:15 2017

@author: root
"""
import tensorflow as tf
from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D, concatenate, Reshape, multiply
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K

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
import random
from dense_all_keras import DenseNetImageNet201

img_rows = 256
img_cols = 256
num_classes = 27

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


    

def generate_processed_batch(inp_data,label,batch_size = 50):

    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_label = np.zeros((batch_size, num_classes))
    while 1:
        for i_data in range(0,len(inp_data),batch_size):
            for i_batch in range(batch_size):
                if i_data + i_batch >= len(inp_data):
                    continue
                try:
                    img = imread(inp_data[i_data+i_batch])
                except Exception:
                    img = imread(inp_data[i_data+i_batch+1])

                try:
                    img = augment(img, np.random.randint(4))
                except Exception:
                    pass
                
                lab = np_utils.to_categorical(label[i_data+i_batch],num_classes)

                try:
                    batch_images[i_batch] = img
                except Exception:
                    print(i_data+i_batch)
                batch_label[i_batch] = lab

            yield batch_images, batch_label


callbacks_list= [
    keras.callbacks.ModelCheckpoint(
        filepath='',
        mode='min',
        monitor='val_loss',
        save_best_only=True,
        verbose = 1
    ),
    EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-3),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, cooldown=1, 
                               verbose=1, min_lr=1e-7)
]

batchsize = 16
#
training_gen = generate_processed_batch(train_imdir, train_label, batchsize)
val_gen = generate_processed_batch(val_imdir,val_label, batchsize)

def get_model():
    base_model = DenseNetImageNet201(input_shape = None,
                        bottleneck=True,
                        reduction=0.5,
                        dropout_rate=0.0,
                        weight_decay=1e-4,
                        include_top=False,
                        weights=None,
                        input_tensor=None,
                        pooling=None,
                        classes=None,
                        )
    x = base_model.output
    x= GlobalAveragePooling2D()(x)

    zz = Dense(10, activation = 'softmax')(x)
    model = Model(inputs = base_model.input, outputs= zz)
    model.load_weights('') #weight of model trained in Phase I
    
    
    model_dresden = Model(inputs = base_model.input, outputs = x)
    p = model_dresden.output
    q = Dense(num_classes, activation = 'softmax')(p)
    
    model_final = Model(inputs = model_dresden.input, outputs= q)
    
    return model_final 


img_rows, img_cols = img_rows, img_rows # Resolution of inputs
channel = 3
num_classes = num_classes
nb_epoch = 50


batch_size=16
model  = get_model()
model.summary()
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(training_gen,steps_per_epoch= len(train_imdir)/batch_size,nb_epoch=nb_epoch,validation_data=val_gen,
                    validation_steps=len(val_imdir)/batch_size,callbacks=callbacks_list,initial_epoch = 1)
