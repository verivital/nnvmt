# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 09:22:25 2019

@author: manzand
"""

import numpy as np
from keras.models import model_from_json
import scipy.io as sio
from tensorflow import keras
#import keras
from keras import models
#from keras.models import load_model
#from keras.models import model_from_json
#import tensorflow as tf
from keras import layers
#from keras.utils import CustomObjectScope
#from keras.initializers import glorot_uniform,glorot_normal

#Load mat file
def load_mat(nn):
    W = []
    b = []
    a = []
    mat = sio.loadmat(nn+'.mat')
    weight = mat['W'].reshape(1,-1)
    bias = mat['b'].reshape(1,-1)
    act = mat['activation_fcns']
    for i in range(bias.size):
        W.append(weight[0][i])
        b.append(bias[0][i].reshape(-1,))
        a.append(act[i].strip()) #Strip to reduce blank spaces in string
        
    return W,b,a
# [W,b,a] = load_mat('CartPolecontroller_0403_tanh.mat')

def create_nn(W,b,a):
    model = models.Sequential()
    model.add(layers.Dense(units = len(W[0]), input_dim=(len(W[0].T)), activation = a[0], weights = [W[0].T,b[0].T.reshape(-1,)]))
    for i in range(len(b)-1):
        model.add(layers.Dense(b[i+1].size, activation = a[i+1], weights = [W[i+1].T,b[i+1].T.reshape(-1,)]))
        
    return model
# model = create_nn(W,b,a)

def save_model(model,nn):
    model.save(nn+'.h5')
    
def parse_model(nn):
    [W,b,a] = load_mat(nn)
    model = create_nn(W,b,a)
    save_model(model,nn)
    

"""
SO FAR, THIS HAS ONLY BEEN TESTED WITH ONE EXAMPLE THAT HAS TANH IN THE HIDDEN 
LAYERS, AND ONE OUTPUT LINEAR LAYER
sECOND EXAMPLE TRIED WAS SUCCESFUL, RESHAPE IS KEY FOR THESE THINGS, MAY NEED TO IMPLEMENT SOME OF THAT
IN THE OTHER PARSERS SO THAT WE ARE CONSISTENT WITH DIMENSIONS 

"""

