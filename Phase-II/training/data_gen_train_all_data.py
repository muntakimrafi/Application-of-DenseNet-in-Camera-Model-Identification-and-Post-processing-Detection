#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 04:17:01 2017

@author: root
"""



train_dir = ''

model = [] #list of models

import random
import glob
import numpy as np
#import cv2, glob, numpy as np, os
from sklearn.feature_extraction import image
#

val_imdir=[]
val_label=[]



train_imdir=[]
train_label=[]




for i in range(len(model)):

    images_all = glob.glob(train_dir + model[i] + '\\*')

    random.shuffle(images_all)
    
    a = len(images_all) * 0.85
    a = np.floor(len(images_all) * 0.85)
    a = int(a)
    b = a %64
    a = a - b
    c = len(images_all)
    d = c %64
    c = c - d
    
    images_all_train = images_all[0:a]
    images_all_val = images_all[a:c]
    

    label_train=[i]*len(images_all_train)
    label_val = [i]*len(images_all_val)

    train_imdir=train_imdir+images_all_train
    train_label=train_label+label_train

    val_imdir=val_imdir+images_all_val
    val_label=val_label+label_val

    
    
    