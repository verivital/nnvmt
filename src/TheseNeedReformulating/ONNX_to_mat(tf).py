# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:01:50 2019

@author: manzand

This file takes as input a neural network in onnx format and converts it to
a mat file using tensorflow.

https://stackoverflow.com/questions/46127471/how-to-get-weights-from-pb-model-in-tensorflow
"""
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
import numpy as np
import scipy.io as sio

#input_path = r'C:\Users\manzand\Documents\MATLAB\ARCH-2019\ACC\controller_3_20.onnx'
#output_path = r'ACC_controller_3_20.mat'

# Load ONNX model
def load_model(input_path):
    onnx_model = onnx.load(input_path)  # load onnx model
    tf_rep = prepare(onnx_model)
    return tf_rep

#Get the dictionary and list of model's tensors
def inform(model):
    #initialize empty lists 
    temp = []
    tensor_list = []
    tensor_dict = model.tensor_dict # Dictionary of tensors
    for key, value in tensor_dict.items(): # Get names of tensors in a list
        temp = [key,value]
        tensor_list.append(temp)
    return tensor_dict,tensor_list

# Get weight, bias and activation functions
def parameters(tensor_dict,tensor_list,model):
    # Get graph and initialize session
    gr = model.graph
    sess = tf.Session(graph = gr)
    #empty lists to store weights, biases and act functions
    W = []
    b = []
    activation_fcns = []
    # After creating session, we will get the weights, activation layers and bias
    for i in range(len(tensor_dict)):
        if 'W' in tensor_list[i][0]:
            W.append(np.float64(sess.run(tensor_dict[tensor_list[i][0]])))
        elif 'B' in tensor_list[i][0]:
            b.append(np.float64(sess.run(tensor_dict[tensor_list[i][0]])))
        elif 'relu' in tensor_list[i][0]:
            activation_fcns.append('relu')
        elif 'linear' in tensor_list[i][0]:
            activation_fcns.append('linear')
        elif 'tanh' in tensor_list[i][0]:
            activation_fcns.append('tanh')
    return W,b,activation_fcns

# Get size parameters of neural network
def net_size(W,b): 
    # Get number of input, output and layer 
    number_of_inputs = np.int64(len(W[0].T))
    number_of_outputs = np.int64(b[-1].size)
    number_of_layers = np.int64(len(b))
    # Get layer sizes
    layer_sizes = []
    for i in range(len(b)):
        layer_sizes.append(np.int64(b[i].size))
    return number_of_inputs,number_of_outputs,number_of_layers,layer_sizes

# Save the nn information in a mat file
def save_nn_mat_file(ni,no,nls,lsize,W,b,lfs,output_path):
    nn1 = dict({'number_of_inputs':ni,'number_of_outputs':no ,'number_of_layers':nls,
                'layer_sizes':lsize,'W':W,'b':b,'activation_fcns':lfs})
    sio.savemat(output_path, nn1)

# parse the nn imported ONNX with tensorflow backend
def parse_nn(input_path,output_path):
    model = load_model(input_path)
    [tensor_dict,tensor_list] = inform(model)
    [W,b,lfs] = parameters(tensor_dict,tensor_list,model)
    [ni,no,nls,lsize] = net_size(W,b)
    save_nn_mat_file(ni,no,nls,lsize,W,b,lfs,output_path)

#parse_nn(input_path,output_path)

