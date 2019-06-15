#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:05:13 2017

@author: root
"""
test_dir=''



import random
import glob, numpy as np, os
from sklearn.feature_extraction import image
#

#
test_imdir=[]
test_label=[]

images = glob.glob(test_dir +'/*')

#
test_imdir=test_imdir+images

    