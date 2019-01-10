#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 20:24:39 2018

@author: Musau
"""


from __future__ import print_function
import argparse
import os




#function that gets input format from the user
def commandLineInterface():
    #create a commmand line tool for NNMT using argparse 
    ap=argparse.ArgumentParser(description="Neural Network Model Translation Tool")
    
    
    ap.add_argument("-i","--input", required=True,
                   help="path to the input file",dest="input")
    ap.add_argument("-o","--output", required=True, help="output path",dest="output")
    ap.add_argument("-t","--tool", required=True, help='input file type i.e (Reluplex,Keras...)',dest="tool")
    ap.add_argument("-f","--format", help='output format to be translated to default: matfile (.mat)',dest="outputFormat",default="mat")
    args=vars(ap.parse_args())#parses the arguments and stores them in a dictionary
    return args


#function that decides which tool the model file is 
def decideTool(name):
    if (name=="Reluplex") or (name=="reluplex"):
        #TO DO: check to see if the input file ends in .nnet
        fileType="Reluplex"
    elif name=="Sherlock" or name=="sherlock":
        #TO DO: check to see if the input file ends in .txt or nothing
        fileType="Sherlock"
    else:
        print("Error: Unrecognized input format. Tools currently supported (Reluplex, Sherlock)")
        fileType=None
    return fileType

#function that decides what the output file should be
def decideOutput(name):
    if name=="mat" or name=="Mat":
        outputType="mat"
    elif name=="onnx" or name=="ONNX":
        outputType="onnx"
    else:
        print("Error: Unrecognized output format. Output formats currently supported (Onnx, Mat)")
        outputType=None
    return outputType

#parse aguments
def parseArguments(arguments):
    inputPath=arguments["input"]
    outputPath=arguments["output"]
    inputFileType=arguments["tool"]
    outputFileType=arguments["outputFormat"]
    
    #decideWhichTool
    a=decideTool(inputFileType)
    #decide which outputFormat
    if(a):
        b=decideOutput(outputFileType)
    #print(inputPath, outputPath, inputFileType, outputFileType)
    
    
    


if __name__=='__main__':
    items=commandLineInterface()
    parseArguments(items)
    

