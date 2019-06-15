# Create your first MLP in Keras
from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,GlobalMaxPooling2D
from keras import backend as K
import numpy as np
from keras.engine.topology import Input
from keras.models import Model
from keras.optimizers import SGD

batch_size = 64
num_classes = 10
epochs = 50

# input image dimensions
img_rows, img_cols = 3, 1920

# the data, split between train and test sets
x_train = np.load('model_64_128_256_feat.npy', 'r')
y_train = np.load('l2_model_label.npy', 'r')

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
#
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)



inp = Input(shape = input_shape)

x =Conv2D(1,(3,1),strides = (3,1),activation = 'relu')(inp)
x = Flatten()(x)

x = Dense(512, activation = 'relu')(x)

x = Dense(256, activation = 'relu')(x)

zz = Dense(10, activation = 'softmax')(x)
model = Model(inputs = inp, outputs= zz)
model.summary()


sgd = SGD(lr=1e-03, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd , loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs)