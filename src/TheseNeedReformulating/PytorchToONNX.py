#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network Verification Model Translation Tool (NNVMT)

@author:  
  Diego Manzanas Lopez (diego.manzanas.lopez@vanderbilt.edu)
  
Notes:
    input_size = [batch_size, channels, height, width]
"""
import os
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
from src.NeuralNetParser import NeuralNetParser
# Super Resolution model definition in PyTorch
import torch.nn as nn
import torch.nn.init as init

class PytorchPrinterCNN(NeuralNetParser):
    def __init__(self,pathToOriginalFile, OutputFilePath, torch_model, input_size, *vals):
        #get the name of the file without the end extension
        import torch_model
        filename=os.path.basename(os.path.normpath(pathToOriginalFile))
        filename=filename.replace('.pth','')
        #save the filename and path to file as a class variable
        self.originalFilename=filename
        self.pathToOriginalFile=pathToOriginalFile
        self.originalFile=open(pathToOriginalFile,"r")
        self.outputFilePath=OutputFilePath
        #if a json file was not specified use the first style parser 
        #otherwise use the second style of parser
        if not vals:
            self.no_json=True
        else:
            self.no_json=False
            self.jsonFile=vals[0]
            
