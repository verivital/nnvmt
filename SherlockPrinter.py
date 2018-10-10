#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 13:10:52 2018

@author: musaup
"""

import numpy as np
import os

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
        network_weights=mat_contents['W']
        network_biases=mat_contents['b']
        del mat_contents['W']
        del mat_contents['b']
        for item in mat_contents.keys():
            nn_info[item]=mat_contents[item]
    else:
        print("File specified is not a .mat file. Please check the directory path")
        return None
    return nn_info, network_weights, network_biases

def create_sherlock_file(source_dir,sourcefile, destination, filename):
    nn_info, network_weights, network_biases=load_nn_mat_file(source_dir,sourcefile)
    file=open(os.path.join(destination,filename)+'.txt','w')
    numberOfInputs=network_weights[0].shape[1]
    numberOfLayers=len(network_weights)-1
    if(len(network_weights[len(network_weights)-1].shape)>1):
        numberOfOutputs=network_weights[len(network_weights)-1].shape[0]
    else:
        numberOfOutputs=1
    file.write(str(numberOfInputs)+'\n')
    file.write(str(numberOfOutputs)+'\n')
    file.write(str(numberOfLayers)+'\n')
    for item in range(0,numberOfLayers):
        file.write(str(network_weights[item].shape[0])+'\n')
    #for item in range(0,numberOfLayers-1):
        #print(network_weights[item].shape[0])
    for layer in range(0,numberOfLayers+1):
        for row in range(0,len(network_weights[layer])):
            if(np.isscalar(network_weights[layer][row])):
                length=1
                for weight in range(0,length):
                    file.write(str(network_weights[layer][row])+'\n')
                file.write(str(network_biases[layer])+'\n')
            else:
                length=len(network_weights[layer][row])
                for weight in range(0,length):
                    file.write(str(network_weights[layer][row][weight])+'\n')
                file.write(str(network_biases[layer][row])+'\n')
    file.close()


#example way to parse matfile into sherlock format    
PATH="/home/musaup/Documents/Research/Examples"
sourcefile='1L_abalone_nets_ready.mat'
DEST='/home/musaup/Documents/Research/Examples'    
create_sherlock_file(PATH,sourcefile,DEST,'1L_abalone_nets_ready')