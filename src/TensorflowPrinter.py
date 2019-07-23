# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:49:27 2019

@author: manzand
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import os
from src.NeuralNetParser import NeuralNetParser
import scipy.io as sio
from onnx import *
import tensorflow as tf
from tensorflow.python.training import checkpoint_utils as cp

# May need to uncomment it if run into errors when running multiple tf sessions
# tf.reset_default_graph()  

#abstract class for keras printers
class TensorflowPrinter(NeuralNetParser):
    #Instantiate files and create a matfile
    def __init__(self,pathToOriginalFile, OutputFilePath, *vals):
        #get the name of the file without the end extension
        filename=os.path.basename(os.path.normpath(pathToOriginalFile))
        filename=filename.replace('.meta','')
        #save the filename and path to file as a class variable
        self.originalFilename=filename
        self.pathToOriginalFile=pathToOriginalFile
        self.originalFile=open(pathToOriginalFile,"r")
        self.outputFilePath=OutputFilePath
        self.parse_nn(filename)
            

    #function for creating the matfile
    def create_matfile(self):
        pass
    #function for creating an onnx model
    def create_onnx_model(self):
        print("Sorry this is still under development")
        
    # load the parameters and models
    def load_network(self,filename):
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(filename+'.meta')
            new_saver.restore(sess, tf.train.latest_checkpoint('./'))
            w = tf.trainable_variables() # create list of weights and biases (tf variables)
            w = sess.run(w) # convert the list of tf variables of w to np arrays
            w1 = sess.graph.get_operations() #gets all the operations done in the network
    
        return w,w1
    #[w,w1] = load_network(filename)
    
    
    def get_layers(self,w,w1):
        lys = []
        for i in range(len(w1)):
            if 'Sigmoid' == w1[i].type:
                lys.append(w1[i].type)
            elif 'Relu' == w1[i].type:
                lys.append(w1[i].type)
            elif 'Tanh' == w1[i].type:
                lys.append(w1[i].type)
            elif 'Softmax' == w1[i].type:
                lys.append(w1[i].type)
    
        return lys
    #lys = get_layers(w,w1)
    
    def get_parameters(self,w):
        W = [] # weights
        b = [] # bias
        lsize = []
        no = len(w[-1])
        if len([w[1].size]) == 1:
            if len(w[0]) == len(w[1]):
                ni = len(w[0][0])
            else:
                ni = len(w[0])
        else:
            if len(w[0]) == len(w[1]):
                ni = len(w[0][0])
            else:
                ni = len(w[0])
        for i in range(int(len(w)/2)):
            W.append(np.float64(w[2*i]))
            b.append(np.float64(w[2*i+1]))
            lsize.append(b[i].size)
        nl = len(b)
            
        return W,b,lsize,ni,no,nl
    
    # Save the neural network to mat file
    def save_nnmat_file(self,ni,no,nl,lsize,W,b,lys):
        nn1 = dict({'W':W,'b':b,'act_fcns':lys})
        sio.savemat(os.path.join(self.outputFilePath, self.originalFilename+".mat"),  nn1)
    #save_nnmat_file(ni,no,nl,n,lsize,W,b,lys)
    
    # parse the nn imported from keras as json and h5 files
    def parse_nn(self,filename):
        [w,w1] = self.load_network(filename)
        lys = self.get_layers(w,w1)
        [W,b,lsize,ni,no,nl] = self.get_parameters(w)
        self.save_nnmat_file(ni,no,nl,lsize,W,b,lys)

    
    # sess.close()
