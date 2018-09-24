#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:39:55 2018

Parser for Sherlock Verification Tool
@author: musaup
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import os

PROJECT_ROOT_DIR='/home/musaup/Documents/Tools/sherlock/OnlyNN'
PROJEC_SAVE_DIR='/home/musaup/Documents/tests/tests2'
FILE_NAME='neural_network_information_15'

def open_nnet_file(project_root_dir,file_name):
    path=os.path.join(project_root_dir,file_name)
    print("opening file:", file_name)
    file=open(path,"r")
    return file

def get_network_info(record):
    #check to see if the file is structures correctly
    dicti={}
    number_of_inputs=record.readline().strip("\n")
    if (np.isscalar(number_of_inputs)):
        number_of_outputs=record.readline().strip("\n")
        number_of_layers=record.readline().strip("\n")
        number_of_neurons_per_layer=record.readline().strip("\n")
        dicti['number_of_inputs']= int(number_of_inputs)
        dicti['number_of_outputs']=int(number_of_outputs)
        dicti['number_of_layers']=int(number_of_layers)
        dicti['number_of_neurons_in_first_layer']= int(number_of_neurons_per_layer)
    else:
        print("Sherlock file not structured correctly")
    return dicti

def create_nn_matrices(info_dict,record):
    numberOfLayers=info_dict['number_of_layers']
    numberOfInputs=info_dict['number_of_inputs']
    numberOfNeurons=info_dict['number_of_neurons_in_first_layer']
    numberOfOutputs=info_dict['number_of_outputs']
    layerSizes=[numberOfInputs]
    for item in range(0,numberOfLayers):
        layerSizes.append(numberOfNeurons)
    #create layer and weight matrix structure
    info_dict['layer_sizes']=layerSizes
    NN_matrix=[0]*(numberOfLayers+1)
    for layers in range(0,numberOfLayers):
        NN_matrix[layers]=[0]*2
        NN_matrix[layers][0]=np.zeros((layerSizes[layers+1],layerSizes[layers]))
        NN_matrix[layers][1]=np.zeros((layerSizes[layers+1],1))
    #create tthe output layer matrix 
    NN_matrix[numberOfLayers]=[0]*2
    NN_matrix[numberOfLayers][0]=np.zeros((numberOfNeurons,numberOfOutputs))
    NN_matrix[numberOfLayers][1]=np.zeros((numberOfOutputs,1))
    
    #fill the NN_matrix
    for layer in range(0,numberOfLayers+1):
        weight_matrix_shape=NN_matrix[layer][0].shape
        bias_matrix_shape=NN_matrix[layer][1].shape
        for index in range(0,weight_matrix_shape[0]):
            for index2 in range(0,weight_matrix_shape[1]):
                line=record.readline().strip("\n")
                NN_matrix[layer][0][index][index2]=float(line)
        for index3 in range(0,bias_matrix_shape[0]):
            for index4 in range(0,bias_matrix_shape[1]):
                line=record.readline().strip("\n")
                NN_matrix[layer][1][index3][index4]=float(line)
    return NN_matrix

def create_nn_matrices_gen(info_dict,record):
    numberOfLayers=info_dict['number_of_layers']
    numberOfInputs=info_dict['number_of_inputs']
    numberOfNeuronsFirstLayer=info_dict['number_of_neurons_in_first_layer']
    numberOfOutputs=info_dict['number_of_outputs']
    layerSizes=[numberOfInputs,numberOfNeuronsFirstLayer]
    for layer in range(0,numberOfLayers-1):
        line=record.readline().strip("\n")
        layerSizes.append(int(line))
    layerSizes.append(numberOfOutputs)
    info_dict['layer_sizes']=layerSizes
    NN_matrix=[0]*(numberOfLayers+1)
    for layers in range(0,numberOfLayers+1):
        NN_matrix[layers]=[0]*2
        NN_matrix[layers][0]=np.zeros((layerSizes[layers+1],layerSizes[layers]))
        NN_matrix[layers][1]=np.zeros((layerSizes[layers+1],1))
    
    #fill the NN_matrix
    for layer in range(0,numberOfLayers+1):
        weight_matrix_shape=NN_matrix[layer][0].shape
        bias_matrix_shape=NN_matrix[layer][1].shape
        for index in range(0,weight_matrix_shape[0]):
            for index2 in range(0,weight_matrix_shape[1]):
                line=record.readline().strip("\n")
                NN_matrix[layer][0][index][index2]=float(line)
        for index3 in range(0,bias_matrix_shape[0]):
            for index4 in range(0,bias_matrix_shape[1]):
                #print(index3,",",index4)
                line=record.readline().strip("\n")
                NN_matrix[layer][1][index3][index4]=float(line)
    return NN_matrix
            
    
            

def create_matfile_matrix_dict(NN_matrix):
    adict={}
    numberOfLayers=len(NN_matrix)
    layerName="layer_"
    layerName1="_weight_matrix"
    biasName="_bias"
    for layer in range(0,numberOfLayers):
        layerKey=layerName+str(layer+1)+layerName1
        biasKey=layerName+str(layer+1)+biasName
        adict[layerKey]=NN_matrix[layer][0]
        adict[biasKey]=NN_matrix[layer][1]
    return adict

def save_mat_file(NN_matrix_dict,info_dict,directory_name,file_name):
    import scipy.io as sio
    import os
    for item in info_dict:
        NN_matrix_dict[item]=info_dict[item]
    path=os.path.join(directory_name,file_name+".mat")
    sio.savemat(path,NN_matrix_dict)
    
def decide_which_file_type(record):
    info_dict=get_network_info(record)
    line=record.readline().strip("\n")
    try:
        a=int(line)
        record.close()
        return isinstance(a,int)
    except ValueError:
        record.close()
        return False

def create_nn_mat_file(nn_source_directory, nn_source_filename, nn_dest_directory,mat_filename):
    record=open_nnet_file(nn_source_directory,nn_source_filename)
    file_type=decide_which_file_type(record)
    if(file_type):
        record=open_nnet_file(nn_source_directory,nn_source_filename)
        info_dict=get_network_info(record)
        nn_mat=create_nn_matrices_gen(info_dict,record)
    else:
        record=open_nnet_file(nn_source_directory,nn_source_filename)
        info_dict=get_network_info(record)
        nn_mat=create_nn_matrices(info_dict,record)
    nn_dict=create_matfile_matrix_dict(nn_mat)
    save_mat_file(nn_dict,info_dict,nn_dest_directory,mat_filename)
    
def parse_all_files(nn_source_directory,nn_destination_directory):
    import os
    path=os.path.join(nn_source_directory)
    if(not os.path.exists(path)):
        print ("That directory does not exist")
    else:
        if(not os.path.exists(nn_destination_directory)):
            os.makedirs(nn_destination_directory)
        dirs=os.listdir(path)
        for item in dirs:
            create_nn_mat_file(path,item,nn_destination_directory,item)

parse_all_files(PROJECT_ROOT_DIR,PROJEC_SAVE_DIR)

def load_nn_mat_file(nn_source_dir,file_name):
    import scipy.io as sio
    import os
    if(file_name.endswith(".mat")):
        path=os.path.join(nn_source_dir,file_name)
        mat_contents=sio.loadmat(path,squeeze_me=True)
        del mat_contents['__header__']
        del mat_contents['__version__']
        del mat_contents['__globals__']
        
        layer_biases=[]
        layer_matrix=[]
        nn_info=[]
        for item in mat_contents:
            if(item.endswith('bias')):
                layer_biases.append(item)
            elif (item.endswith('matrix')):
                layer_matrix.append(item)
            else:
                nn_info.append(item)
        layer_biases.sort()
        layer_matrix.sort()     
    else:
        print("File specified is not a .mat file. Please check the directory path")
        return None
    return nn_info, layer_matrix, layer_biases, mat_contents

def load_nn_model_from_mat_file_sherlock(nn_source_dir,file_name):
    nn_info, layer_matrix, layer_biases, mat_contents=load_nn_mat_file(nn_source_dir,file_name)
    labels=['number_of_inputs','number_of_outputs','number_of_layers','number_of_neurons_in_first_layer','layer_sizes']
    numberOfLayers=mat_contents['number_of_layers']
    NN_matrix=[0]*numberOfLayers
    from collections import OrderedDict
    NN_info_dict=OrderedDict()
    for i in range(0,len(NN_matrix)):
        NN_matrix[i]=[0]*2
    for layer in range(0,len(NN_matrix)):
            NN_matrix[layer][0]=mat_contents[layer_matrix[layer]]
            NN_matrix[layer][1]=mat_contents[layer_biases[layer]]
    for label in labels:
        NN_info_dict[label]=mat_contents[label]
    return NN_info_dict, NN_matrix

#print information in nn_info dictionary
def print_nn_info(nn_info_dict):
    for item in nn_info_dict:
        if(np.isscalar(nn_info_dict[item])):
            print(item+":",nn_info_dict[item])
        else:
            print(item+":",np.array_str(nn_info_dict[item],precision=4))
    print("---------------------------------------")


#load all of the model files in a directory    
def load_all_model_files(nn_source_dir):
    import os
    path=os.path.join(nn_source_dir)
    if(not os.path.exists(path)):
        print("That directory does not exist")
        return None
    else:
        dirs=os.listdir(nn_source_dir)
        dirs2=[]
        for item in dirs:
            if (item.endswith('.mat')):
                dirs2.append(item)
        length=len(dirs2)
        if(length==0):
            print("There are no mat files in that directory")
            return []
        Neural_Networks=[0]*length
        for net in range(0,length):
            Neural_Networks[net]=[0]*2
        for file_name in range(0,length):
            NN_info,NN_matrix=load_nn_model_from_mat_file_sherlock(nn_source_dir,dirs2[file_name])
            Neural_Networks[file_name][0]=NN_info
            Neural_Networks[file_name][1]=NN_matrix
    return Neural_Networks

#print all of the information of the networks in a directory
def print_loaded_networks(list_of_networks):
    length=len(list_of_networks)
    for i in range(0,length):
        print_nn_info(list_of_networks[i][0])

NN_SOURCE=PROJEC_SAVE_DIR
net_list=load_all_model_files(NN_SOURCE)
print_loaded_networks(net_list)