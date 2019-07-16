#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 13:50:54 2018

@author: musaup
"""

from __future__ import division, print_function, unicode_literals, absolute_import
import numpy as np
import os
from src.NeuralNetParser import NeuralNetParser
from onnx import *
import scipy.io as sio

class sherlockPrinter(NeuralNetParser):
    
    def __init__(self,pathToOriginalFile, OutputFilePath,*vals):
        filename=os.path.basename(os.path.normpath(pathToOriginalFile))
        filename=filename.replace('.txt','')
        self.originalFilename=filename
        self.pathToOriginalFile=pathToOriginalFile
        self.originalFile=open(pathToOriginalFile,"r")
        self.outputFilePath=OutputFilePath
        self.network_weight_matrices, self.network_bias_matrices,self.info_dict=self.create_matfile()
        self.originalFile.close()
        

    #function to save the matfiles with the network weights
    def saveMatfile(self):
        self.save_mat_file(self.info_dict,self.network_weight_matrices,self.network_bias_matrices,self.outputFilePath, self.originalFilename)
        
    #create an onnx model using the matfiles created by create_matfile  
    def  create_onnx_model(self):
        #TO DO IMPLEMENT THIS
        model_def=self.createSherlockOnnx(self.network_weight_matrices,self.network_bias_matrices,self.info_dict['layer_sizes'])
        new_model_path = os.path.join(self.outputFilePath, self.originalFilename)
        onnx.save(model_def, new_model_path+".onnx")
        
        
    

    def create_matfile(self):
        record=self.originalFile
        self.file_type=self.decide_which_file_type(record)
        if(self.file_type):
            record=open(self.pathToOriginalFile,"r")
            info_dict=self.get_network_info(record)
            nn_mat=self.create_nn_matrices_gen(info_dict,record)
        else:
            record=open(self.pathToOriginalFile,"r")
            info_dict=self.get_network_info(record)
            nn_mat=self.create_nn_matrices(info_dict,record)
        network_weight_matrices, network_bias_matrices=self.create_matfile_matrix_dict(nn_mat)
        return network_weight_matrices, network_bias_matrices, info_dict
    
    #function that handles the creation of a sherlock Onnx model
    def createSherlockOnnx(self,W,b,layerSizes):
        num_layers=len(W)
        layer_name="FC"
        input_name="X"
        inputSize=layerSizes[0]
        if(self.file_type):
            outputSize=layerSizes[len(W)]
        else:
            outputSize=layerSizes[len(W)-1]
        output_name="Y"
        reluOutput="R"
        output1="O"
        reluOperation="ReLU"
        weightLabels, biasLabels=self.createWeightBiasLabels(W)
        nodeList=[]
        tensorList=[]
        if(int(inputSize)==1):
             tensorList.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [1]))
        else:
            tensorList.append(helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [int(inputSize),1]))
        if(outputSize==1):
            outputList=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1])]
        else:
            outputList=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [int(outputSize),1])]
        for i in range(0,num_layers):
            nodeList.append(helper.make_node(layer_name, [input_name, weightLabels[i], biasLabels[i]], [output1+str(i+1)]))
            input_name=output1+str(i+1)
            if(i==num_layers-1):
                nodeList.append(helper.make_node(reluOperation, [input_name], [output_name]))
            else:
                nodeList.append(helper.make_node(reluOperation, [input_name], [reluOutput+str(i+1)]))
            input_name=reluOutput+str(i+1)
        for i in range(0,num_layers):
            weightMatrix=W[i]
            biasMatrix=b[i]
            if(weightMatrix.shape[0]==1 and weightMatrix.shape[1]==1):
                tensorList.append(helper.make_tensor_value_info(weightLabels[i], TensorProto.FLOAT, [1]))
            else:
                tensorList.append(helper.make_tensor_value_info(weightLabels[i], TensorProto.FLOAT, [weightMatrix.shape[0],weightMatrix.shape[1]]))
            if(biasMatrix.shape[0]==1 and biasMatrix.shape[1]==1):
                tensorList.append(helper.make_tensor_value_info(biasLabels[i], TensorProto.FLOAT, [1]))
            else:
                tensorList.append(helper.make_tensor_value_info(biasLabels[i], TensorProto.FLOAT, [biasMatrix.shape[0],biasMatrix.shape[1]]))
        graph=helper.make_graph(nodeList,"MLP",tensorList,outputList)
        model_def = helper.make_model(graph, producer_name='Sherlock->Onnx')
        return model_def

    def createWeightBiasLabels(self,W):
        weightNames=[]
        biasNames=[]
        weightPrefix="W"
        biasPrefix="B"
        for i in range(0,len(W)):
            weightNames.append(weightPrefix+str((i+1)))
            biasNames.append(biasPrefix+str((i+1)))
        return weightNames, biasNames
        
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
            act_array=["relu"]*(int(number_of_layers)+1)
            act_array[-1]="linear"
            dicti["act_fcns"]=act_array
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
        NN_matrix[numberOfLayers][0]=np.zeros((numberOfOutputs,numberOfNeurons))
        NN_matrix[numberOfLayers][1]=np.zeros((numberOfOutputs,1))
    
        #fill the NN_matrix
        for layer in range(0,numberOfLayers+1):
            weight_matrix_shape=NN_matrix[layer][0].shape
            bias_matrix_shape=NN_matrix[layer][1].shape
            for index in range(0,weight_matrix_shape[0]):
                for index2 in range(0,weight_matrix_shape[1]):
                    line=record.readline().strip("\n")
                    NN_matrix[layer][0][index][index2]=float(line)
                line=record.readline().strip("\n")
                NN_matrix[layer][1][index][0]=float(line)
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
                line=record.readline().strip("\n")
                NN_matrix[layer][1][index][0]=float(line)
        return NN_matrix
    
    def create_matfile_matrix_dict(self, NN_matrix):
        numberOfLayers=len(NN_matrix)
        network_weight_matrices=np.zeros((numberOfLayers,), dtype=np.object)
        network_bias_matrices=np.zeros((numberOfLayers,), dtype=np.object)
        for layer in range(0,numberOfLayers):
            network_weight_matrices[layer]=NN_matrix[layer][0]
            network_bias_matrices[layer]=NN_matrix[layer][1]
        return network_weight_matrices, network_bias_matrices

    def save_mat_file(self,info_dict,network_weight_matrices,network_bias_matrices,directory_name,file_name):
        NN_matrix_dict={}
        import scipy.io as sio
        import os
        NN_matrix_dict["act_fcns"]=info_dict["act_fcns"]
        NN_matrix_dict['W']=network_weight_matrices
        NN_matrix_dict['b']=network_bias_matrices
        path=os.path.join(directory_name,file_name+".mat")
        sio.savemat(path,NN_matrix_dict)
        return NN_matrix_dict
        
    
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
