#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 20:36:27 2018

@author: Musau
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import tensorflow as tf
import os 
import numpy as np
from tensorflow import keras

mnist=tf.keras.datasets.mnist

# To plot pretty figures
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

print(tf.__version__)

(X_train, y_train), (x_test, y_test)= mnist.load_data()
X_train=X_train/255
x_test=x_test/255

X_train[0]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
    
    
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(500,activation=tf.nn.relu),
    keras.layers.Dense(500,activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.relu)
])
    
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(metrics=['accuracy'],loss='mse',optimizer=adam)

model.fit(X_train, y_train, epochs=20)