# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 20:56:08 2018

@author: User
"""
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
import numpy as np
from keras.applications.densenet import DenseNet201
from imageio import imread

img_rows = 256
img_cols = 256
channel = 3

all_feature = np.zeros((len(train_imdir),1920))

def get_model():
    
    base_model = DenseNet201(include_top=False, weights='imagenet', input_shape= (img_rows, img_cols, channel), pooling=None, classes=None)   
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    zz = Dense(10, activation = 'softmax')(x)
    
    model = Model(inputs = base_model.input, outputs= zz)
    return model

model =get_model()
model.load_weights('')
model.summary()

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(index = -2).output)

for i in range(len(train_imdir)):
    img = np.expand_dims(imread(train_imdir[i]), axis = 0)
    all_feature[i] = intermediate_layer_model.predict(img)[0]
    if(i%10==0):
        print(i)