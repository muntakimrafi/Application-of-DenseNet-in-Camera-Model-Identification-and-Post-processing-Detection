# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 13:05:52 2018

@author: DSP
"""
import numpy as np
import random, glob, os
from tqdm import tqdm

# directory of the input image
core_dir =  ""

processings = [ name.split(os.sep)[-1] for name in glob.glob(core_dir)]
camera_models = [ name.split(os.sep)[-1] for name in glob.glob(glob.glob(core_dir)[0]+'\\*')]

train_dir, train_label, val_dir, val_label = [],[],[],[]

split_ratio = np.array([0.85, 0.15]); #train, valid, test; sum must be 1.0

batch_size = 4

for label in tqdm(range(len(camera_models))):
    model_images = glob.glob(core_dir + '\\' + camera_models[label] +'\\*')
    splt = np.cumsum(batch_size*((len(model_images)*split_ratio).astype(int)//batch_size))
    train_dir = train_dir + model_images[0:splt[0]]
    train_label = train_label + [label]*len(model_images[0:splt[0]])
    val_dir = val_dir + model_images[splt[0]:splt[1]]
    val_label = val_label + [label]*len(model_images[splt[0]:splt[1]])
    test_dir = test_dir + model_images[splt[1]:splt[2]]
    test_label = test_label +  [label]*len(model_images[splt[1]:splt[2]])
    label += 1
    
train_data = list(zip(train_dir, train_label))
valid_data = list(zip(val_dir, val_label))
test_data = list(zip(test_dir, test_label))

del train_dir, train_label, val_dir, val_label, test_dir, test_label 
del label, model_images, split_ratio, splt

#%%
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, merge, ZeroPadding2D, concatenate
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import np_utils
from dense_all_keras import DenseNetImageNet201
from sklearn.metrics import log_loss
from imageio import imread
#%%
random.shuffle(train_data)
random.shuffle(valid_data)

img_rows = 256
img_cols = 256
n_cls = len(camera_models)
nb_epoch = 50

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


def generate_processed_batch(data,batch_size = 16):
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_label = np.zeros((batch_size, n_cls))
    
    while 1:
        random.shuffle(data)      
        for i_data in range(0,len(data),batch_size):
            for i_batch in range(batch_size):
                img = imread(data[i_data+i_batch][0])
                img = augment(img, np.random.randint(4))
                limit = 256 - img_rows
                corner = np.random.randint(0,limit)
                img = img[corner:corner+img_rows, corner:corner+img_rows,:]
                lab = np_utils.to_categorical(data[i_data+i_batch][1],n_cls)
                batch_images[i_batch] = img
                batch_label[i_batch] = lab

            yield batch_images, batch_label

def generate_processed_batch_val(data,batch_size = 16):
    batch_images = np.zeros((batch_size, img_rows, img_cols, 3))
    batch_label = np.zeros((batch_size, n_cls))
    
    while 1:     
        for i_data in range(0,len(data),batch_size):
            for i_batch in range(batch_size):
                img = imread(data[i_data+i_batch][0])
                img = augment(img, np.random.randint(4))
                limit = 256 - img_rows
                corner = np.random.randint(0,limit)
                img = img[corner:corner+img_rows, corner:corner+img_rows,:]
                lab = np_utils.to_categorical(data[i_data+i_batch][1],n_cls)
                batch_images[i_batch] = img
                batch_label[i_batch] = lab

            yield batch_images, batch_label
        
            


callbacks_list= [
    keras.callbacks.ModelCheckpoint(
        filepath='desne_process_re.h5',
        mode='min',
        monitor='val_loss',
        save_best_only=True,
        verbose = 1
    ),
    EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-3),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, cooldown=1, 
                               verbose=1, min_lr=1e-7)
]
    
    
training_gen = generate_processed_batch(train_data, batch_size)
val_gen = generate_processed_batch_val(valid_data, batch_size)

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

zz = Dense(n_cls, activation = 'softmax')(x)
model = Model(inputs = base_model.input, outputs= zz)
    
    
model.summary()

#%%
sgd = SGD(lr=1e-03, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
#%%
model.fit_generator(training_gen,
                    steps_per_epoch=int(len(train_data)/batch_size),
                    nb_epoch=nb_epoch,
                    validation_data=val_gen,
                    validation_steps=int(len(valid_data)/batch_size),
                    callbacks=callbacks_list,
                    initial_epoch=0 )