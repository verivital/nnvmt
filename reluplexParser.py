#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 14:43:53 2018

@author: musaup
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import os

PROJECT_ROOT_DIR='/home/musaup/Documents/Tools/ReluplexCav2017/nnet'
#FILE_NAME='ACASXU_run2a_1_1_batch_2000.nnet'
FILE_NAME='ACASXU_run2a_4_3_batch_2000.nnet'

def open_nnet_file(project_root_dir,file_name):
    path=os.path.join(project_root_dir,file_name)
    print("opening file:", file_name)
    file=open(path,"r")
    return file

#function that skips all of the comments in the file
def skip_comments(record):
    line=record.readline()
    commentline=line.find("/")
    while(commentline>=0):
        line=record.readline()
        commentline=line.find("/")
    return line

def process_network_information(first,record):
    #process first line which gives layers, inputs, outputs, and the sizeOfLargestLayer
    first_line=first.split(",")
    numberOfLayers=int(first_line[0])
    print("Number Of Layers: ",(numberOfLayers))
    numberOfInputs=int(first_line[1])
    print("Number of Inputs: ",numberOfInputs)
    numberOfOutputs=int(first_line[2])
    print("Number of Outputs: ",numberOfOutputs)
    sizeOfLargestLayer=int(first_line[3])
    print("sizeOfLargestLayer: ",sizeOfLargestLayer)
    
    #create an array that has all of the layer sizes
    line=record.readline()
    second_line=line.split(",")
    layer_sizes=np.zeros(len(second_line)-1)
    for i in range(0,(len(second_line)-1)):
        layer_sizes[i]=int(second_line[i])
    layer_sizes=layer_sizes.astype(int)
    
    print("Layer Sizes: ",layer_sizes)
    #determine if the network is symmetric or not
    line=record.readline()
    symmetric=line[0]
    print("Network is symmetric: ",symmetric==1)
    #store minimum and maximums of inputs and arrays
    line=record.readline()
    fourth_line=line.split(",")
    
    #Minimums
    MIN=np.zeros(len(fourth_line)-1)
    for i in range(0,(len(fourth_line)-1)):
        MIN[i]=fourth_line[i]
    print("Minimum of Inputs",MIN)
    line=record.readline()
    fifth_line=line.split(",")
    
    
    #maximums
    MAX=np.zeros(len(fifth_line)-1)
    for i in range(0,(len(fifth_line)-1)):
        MAX[i]=fifth_line[i]
    print("Maximums of inputs: ",MAX)
    
    #Load Mean and Range of Inputs
    line=record.readline()
    sixth_line=line.split(",")
    
    #mean
    mean=np.zeros(len(sixth_line)-1)
    for i in range(0,len(sixth_line)-1):
        mean[i]=sixth_line[i]
    line=record.readline()
    seventh_line=line.split(",")
    range1=np.zeros(len(seventh_line)-1)
    for i in range(0,len(sixth_line)-1):
        range1[i]=seventh_line[i]
    print("Means used to Scale Inputs and Outputs: ",mean)
    print("Range used to Scale Inputs and Outputs: ",range1)
    print("------------------------------------------------------------")
    return line, numberOfLayers, layer_sizes, sizeOfLargestLayer, MIN, MAX, mean, range1,symmetric


def create_nn_matrix(numberOfLayers,layer_sizes):
    NN_matrix=[0]*numberOfLayers
    for i in range(0,len(NN_matrix)):
        NN_matrix[i]=[0]*2
    for layer in range(0,numberOfLayers):
        NN_matrix[layer][0]=np.zeros((layer_sizes[layer+1],layer_sizes[layer]))
        NN_matrix[layer][1]=np.zeros((layer_sizes[layer+1],1))
    return NN_matrix

#create dictionary for weights and biases so that it can be saved as a mat file
def save_mat_file(NN_matrix,numberOfLayers):
    weightsdictionary=[]
    biasdictionary=[]
    dictionary={}
    for layer in range(0,numberOfLayers):
        weightsdictionary.append(NN_matrix[layer][0])
        biasdictionary.append(NN_matrix[layer][1])
    dictionary["W"]=weightsdictionary
    dictionary["b"]=biasdictionary
    return dictionary

#create dictionary that stores the basic information of the network so that it can be stored as a mat file
def save_nn_info(dictionary,numberOfLayers, sizeOfLargestLayer, MIN, MAX, mean, range1,symmetric):
    labels=['size_Of_Largest_Layer','Minimum_of_Inputs','Maximum_of_Inputs','means_for_scaling','range_for_scaling','is_symmetric']
    items=[sizeOfLargestLayer, MIN, MAX, mean, range1,symmetric]
    for item in range(0,len(labels)):
        dictionary[labels[item]]=items[item]
    return dictionary

def fill_NN_matrix(NN_matrix,record,numberOfLayers,layer_sizes):
    for layer in range(0,numberOfLayers):
        for index in range(0,2):
            for index2 in range(0,layer_sizes[layer+1]):
                line=record.readline()
                split_array=line.split(",")
                split_array.remove('\n')
                for index3 in range(0,len(split_array)):
                    NN_matrix[layer][index][index2][index3]=split_array[index3]
    return NN_matrix

#function that creates matfile for .nnnet file given the root path, the filename, specify the directory where to store it
# and what you want to call it
def create_nn_mat_file(nn_project_root_dir,nn_file_name,mat_file_dir,mat_filename):
    record=open_nnet_file(nn_project_root_dir,nn_file_name)
    first_line=skip_comments(record)
    line,numberOfLayers,layer_sizes, sizeOfLargestLayer, MIN, MAX, mean, range1,symmetric=process_network_information(first_line,record)
    #print(numberOfLayers)
    NN_matrix=create_nn_matrix(numberOfLayers,layer_sizes)
    #print(NN_mat[0])
    NN_matrix=fill_NN_matrix(NN_matrix,record,numberOfLayers,layer_sizes)
    
    from collections import OrderedDict
    adict=OrderedDict()
    
    adict=save_mat_file(NN_matrix,numberOfLayers)
    adict1=save_nn_info(adict,numberOfLayers, sizeOfLargestLayer, MIN, MAX, mean, range1,symmetric)
    adict1["layer_sizes"]=layer_sizes
    path=os.path.join(mat_file_dir,mat_filename+".mat")
    import scipy.io as sio
    sio.savemat(path,adict1)



#parse all files in a directory and specify where you want to save the new mat files
def parse_all_files(nn_source_dir,nn_net_dir,dir_name):
    import os
    path=os.path.join(nn_net_dir,dir_name)
    if(not os.path.exists(path)):
        os.makedirs(path)
    dirs=os.listdir(nn_source_dir)
    dirs2=[]
    for i in dirs:
        if(i.endswith('.nnet')):
            dirs2.append(i)
            print (i)
    for i in dirs2:
        print(i.strip(".nnet"))
        create_nn_mat_file(nn_source_dir,i,path,i.strip(".nnet"))
        
        
#Here is an example of how to do it
file1_dir='/home/musaup/Documents/'
file2_dir='/home/musaup/Documents/Tools/ReluplexCav2017/nnet'
parse_all_files(file2_dir,file1_dir,"tests")


#load a mat file into python
def load_nn_mat_file(nn_source_dir,file_name):
    import scipy.io as sio
    import os
    if(file_name.endswith(".mat")):
        path=os.path.join(nn_source_dir,file_name)
        mat_contents=sio.loadmat(path,squeeze_me=True)
        del mat_contents['__header__']
        del mat_contents['__version__']
        del mat_contents['__globals__']
        nn_info={}
        layer_weight_matrix=mat_contents['W']
        network_biases=mat_contents['b']
        
        del mat_contents['W']
        del mat_contents['b']
        for item in mat_contents.keys():
            if(item=="is_symmetric"):
                nn_info[item]=mat_contents[item]==1
            else:
                nn_info[item]=mat_contents[item]
    else:
        print("File specified is not a .mat file. Please check the directory path")
        return None
    return layer_weight_matrix, network_biases, nn_info


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
        Neural_Networks=[0]*length
        for net in range(0,length):
            Neural_Networks[net]=[0]*3
        for file_name in range(0,length):
            layer_weight_matrix, network_biases, nn_info= load_nn_mat_file(nn_source_dir,dirs2[file_name])
            Neural_Networks[file_name][0]=nn_info
            Neural_Networks[file_name][1]=layer_weight_matrix
            Neural_Networks[file_name][2]=network_biases
    return Neural_Networks
            

#print all of the information of the networks in a directory
def print_loaded_networks(list_of_networks):
    length=len(list_of_networks)
    for i in range(0,length):
        print_nn_info(list_of_networks[i][0])

#NN_SOURCE='/home/musaup/Documents/tests'
#net_list=load_all_model_files(NN_SOURCE)
#print_loaded_networks(net_list)

#mat_file_dir='/home/musaup/Documents/tests'
#create_nn_mat_file(PROJECT_ROOT_DIR,FILE_NAME,mat_file_dir,"W_b_test")
        
#file1_dir='/home/musaup/Documents/'
#file2_dir='/home/musaup/Documents/Tools/ReluplexCav2017/nnet'
#parse_all_files(file2_dir,file1_dir,"testingfunc")

NN_SOURCE='/home/musaup/Documents/tests'
net_list=load_all_model_files(NN_SOURCE)
print_loaded_networks(net_list)