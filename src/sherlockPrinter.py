#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:50:54 2018

@author: musaup
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import os
from src.NeuralNetParser import NeuralNetParser
import scipy.io as sio

class sherlockPrinter(NeuralNetParser):
    
    def __init__(self,pathToOriginalFile, OutputFilePath):
        filename=os.path.basename(os.path.normpath(pathToOriginalFile))
        filename=filename.replace('.txt','')
        self.originalFilename=filename
        self.pathToOriginalFile=pathToOriginalFile
        self.originalFile=open(pathToOriginalFile,"r")
        self.outputFilePath=OutputFilePath
        self.create_matfile()
        
    def load_model(self):
        #TO DO IMPLEMENT THIS
        print("hello")
    def  create_onnx_model(self):
        #TO DO IMPLEMENT THIS
        print("work in progress")
        
    def create_matfile(self):
        record=self.originalFile
        file_type=self.decide_which_file_type(record)
        if(file_type):
            record=open(self.pathToOriginalFile,"r")
            info_dict=self.get_network_info(record)
            nn_mat=self.create_nn_matrices_gen(info_dict,record)
        else:
            record=open(self.pathToOriginalFile,"r")
            info_dict=self.get_network_info(record)
            nn_mat=self.create_nn_matrices(info_dict,record)
        nn_dict=self.create_matfile_matrix_dict(nn_mat)
        self.save_mat_file(nn_dict,info_dict,self.outputFilePath,self.originalFilename)
        
    def get_network_info(self, record):
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
    
    
    def create_nn_matrices(self, info_dict,record):
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
    
    def create_nn_matrices_gen(self, info_dict,record):
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
    
    def create_matfile_matrix_dict(self, NN_matrix):
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

    def save_mat_file(self, NN_matrix_dict,info_dict,directory_name,file_name):
        for item in info_dict:
            NN_matrix_dict[item]=info_dict[item]
        path=os.path.join(directory_name,file_name+".mat")
        sio.savemat(path,NN_matrix_dict)
        
    
    def decide_which_file_type(self, record):
        info_dict=self.get_network_info(record)
        line=record.readline().strip("\n")
        try:
            a=int(line)
            record.close()
            return isinstance(a,int)
        except ValueError:
            record.close()
            return False