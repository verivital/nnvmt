#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:55:05 2018

@author: Musau
"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import os
from src.NeuralNetParser import NeuralNetParser
import scipy.io as sio
from onnx import *

class reluplexPrinter(NeuralNetParser):
    
    
    def __init__(self,pathToOriginalFile, OutputFilePath,*vals):
        filename=os.path.basename(os.path.normpath(pathToOriginalFile))
        filename=filename.replace('.nnet','')
        self.originalFilename=filename
        self.originalFile=open(pathToOriginalFile,"r")
        self.outputFilePath=OutputFilePath
        self.create_matfile()
        
    def load_model(self):
        #TO DO IMPLEMENT THIS
        print("hello")
    def  create_onnx_model(self):
        model_def=self.createReluplexOnnx(self.network_weight_matrices,self.network_bias_matrices,self.layer_sizes)
        new_model_path = os.path.join( self.outputFilePath, self.originalFilename)
        onnx.save(model_def, new_model_path+".onnx")
        
    def saveMatfile(self):
        path=os.path.join(self.outputFilePath,self.originalFilename+".mat")
        sio.savemat(path,self.matDict)
        

    def create_matfile(self):
        record=self.originalFile
        first_line=self.skip_comments(record)
        line,numberOfLayers,self.layer_sizes, sizeOfLargestLayer, MIN, MAX, mean, range1,symmetric=self.process_network_information(first_line,record)
        NN_matrix=self.create_nn_matrix(numberOfLayers,self.layer_sizes)
        NN_matrix=self.fill_NN_matrix(NN_matrix,record,numberOfLayers,self.layer_sizes)
        adict,self.network_weight_matrices, self.network_bias_matrices=self.create_matdict(NN_matrix,numberOfLayers)
        adict1=self.create_nn_info_dict(adict,numberOfLayers, sizeOfLargestLayer, MIN, MAX, mean, range1,symmetric)
        adict1["layer_sizes"]=self.layer_sizes
        self.matDict=adict1
        
        
        
    def createWeightBiasLabels(self,W):
        weightNames=[]
        biasNames=[]
        weightPrefix="W"
        biasPrefix="B"
        for i in range(0,len(W)):
            weightNames.append(weightPrefix+str((i+1)))
            biasNames.append(biasPrefix+str((i+1)))
        return weightNames, biasNames
    
    def createReluplexOnnx(self,W,b,layerSizes):
        num_layers=len(W)
        layer_name="FC"
        input_name="X"
        inputSize=layerSizes[0]
        outputSize=layerSizes[len(W)]
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
            outputList=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [int(outputSize),1])]
        else:
            outputList=[helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1])]
        for i in range(0,num_layers):
            nodeList.append(helper.make_node(layer_name, [input_name, weightLabels[i], biasLabels[i]], [output1+str(i+1)]))
            input_name=output1+str(i+1)
            if(i==num_layers-1):
                continue
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
        model_def = helper.make_model(graph, producer_name='Reluplex->Onnx')
        return model_def

    #define helper functions for this class
    #function that skips all of the comments in the file
    def skip_comments(self,record):
        line=record.readline()
        commentline=line.find("//")
        while(commentline>=0):
            line=record.readline()
            commentline=line.find("//")
        return line
        
    def process_network_information(self,first,record):
        #process first line which gives layers, inputs, outputs, and the sizeOfLargestLayer
        first_line=first.split(",")
        numberOfLayers=int(first_line[0])
        numberOfInputs=int(first_line[1])
        numberOfOutputs=int(first_line[2])
        sizeOfLargestLayer=int(first_line[3])
        #create an array that has all of the layer sizes
        line=record.readline()
        second_line=line.split(",")
        layer_sizes=np.zeros(len(second_line)-1)
        for i in range(0,(len(second_line)-1)):
            layer_sizes[i]=int(second_line[i])
        layer_sizes=layer_sizes.astype(int)
        #determine if the network is symmetric or not
        line=record.readline()
        symmetric=line[0]
        #store minimum and maximums of inputs and arrays
        line=record.readline()
        fourth_line=line.split(",")
        
        #Minimums
        MIN=np.zeros(len(fourth_line)-1)
        for i in range(0,(len(fourth_line)-1)):
            MIN[i]=fourth_line[i]
        line=record.readline()
        fifth_line=line.split(",")
        
        
        #maximums
        MAX=np.zeros(len(fifth_line)-1)
        for i in range(0,(len(fifth_line)-1)):
            MAX[i]=fifth_line[i]
        
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
        return line, numberOfLayers, layer_sizes, sizeOfLargestLayer, MIN, MAX, mean, range1,symmetric
    
    def create_nn_matrix(self,numberOfLayers,layer_sizes):
        NN_matrix=[0]*numberOfLayers
        for i in range(0,len(NN_matrix)):
            NN_matrix[i]=[0]*2
        for layer in range(0,numberOfLayers):
            NN_matrix[layer][0]=np.zeros((layer_sizes[layer+1],layer_sizes[layer]))
            NN_matrix[layer][1]=np.zeros((layer_sizes[layer+1],1))
        return NN_matrix
    
    #create dictionary for weights and biases so that it can be saved as a mat file
    def create_matdict(self,NN_matrix,numberOfLayers):
        weightsdictionary=[]
        biasdictionary=[]
        dictionary={}
        for layer in range(0,numberOfLayers):
            weightsdictionary.append(NN_matrix[layer][0])
            biasdictionary.append(NN_matrix[layer][1])
        dictionary["W"]=weightsdictionary
        dictionary["b"]=biasdictionary
        return dictionary,np.asarray(weightsdictionary),np.asarray(biasdictionary)

    #create dictionary that stores the basic information of the network so that it can be stored as a mat file
    def create_nn_info_dict(self, dictionary,numberOfLayers, sizeOfLargestLayer, MIN, MAX, mean, range1,symmetric):
        labels=['size_Of_Largest_Layer','Minimum_of_Inputs','Maximum_of_Inputs','means_for_scaling','range_for_scaling','is_symmetric']
        items=[sizeOfLargestLayer, MIN, MAX, mean, range1,symmetric]
        for item in range(0,len(labels)):
            dictionary[labels[item]]=items[item]
        return dictionary
    
    def fill_NN_matrix(self,NN_matrix,record,numberOfLayers,layer_sizes):
        for layer in range(0,numberOfLayers):
            for index in range(0,2):
                for index2 in range(0,layer_sizes[layer+1]):
                    line=record.readline()
                    split_array=line.split(",")
                    if '\n' in split_array:
                        split_array.remove('\n')
                    elif '' in split_array:
                        split_array.remove('')
                    if(len(split_array)>1):
                        for index3 in range(0,len(split_array)):
                            NN_matrix[layer][index][index2][index3]=split_array[index3]
                    else:
                        NN_matrix[layer][index][index2]=split_array
        return NN_matrix
    
    


