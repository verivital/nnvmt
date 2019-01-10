# Neural Network Model Translation  (NNMT)

This repository contains an implementation of a translation tool for neural network models into the [Open Neural Network Exchange format](https://github.com/onnx) developped by Facebook and Microsoft. The tool is also able to neural network models so that the weights and biases can be exported into the Microsoft Access Table (.mat) format. We are also working on implementing printing from the ONNX format into the various input formats of the formal sverification software tools available within the research literature. 

## Installing the translator
Make sure you have the following packages installed:
  - Numpy 
  - Scipy
  - Keras
  - ONNX
     - Instructions can be found [here](https://github.com/onnx/onnx)
  - Pathlib
     -'''conda install -c menpo pathlib ''' 
### Linux
- make sure you have installed [TkInter](https://wiki.python.org/moin/TkInter)
  - if you have anaconda run:
        ```conda install -c anaconda tk```
## Parsers available for neural networks created in the following libraries:
- Keras
## Verification Tools Currently Supported
- Reluplex
- Sherlock
## NNMT Usage 
NNMT has been tested on MacOS Mojave Version 10.14
#### GUI 
NNMT can be run through a GUI. To use the GUI, after installing the above libraries simply run `main.py`
#### Command Line Usage
Under development
#### Translating into other model formats
To convert into the formats of other tools such as [Caffe2](https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=prebuilt) [PyTorch] [Matlab](https://www.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format) or [several others](http://onnx.ai/getting-started) use the ONNX converter which can be found [here](https://github.com/onnx/tutorials)

## Repository Organization
- src: contains the code for translating the models
- examples: contains several input format types that one can use to test out the tool
- ONNX: a collection of neural network models stored in the ONNX format
## Contact
For questions please contact 
1. Patrick Musau: patrick.musau@vanderbilt.edu
2. Diego Manzanas: diego.manzanas.lopez@vanderbilt.edu


