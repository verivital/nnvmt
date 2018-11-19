# Neural Network Tool Parsers

This repository contains an implementation of a translation tool for neural network models into the [Open Neural Network Exchange format](https://github.com/onnx) developped by Facebook and Microsoft. The tool is also able to parse feedforward neural network models so that the weights and biases can be exported into the Microsoft Access Table (.mat) format.  

for creating input files in the format of the various neural network verification tools available in the research community. It also contains scripts for translating neural network models into .mat files that store  the network weight matrices and biases for each layer sequentially.


## Installing the translator
Make sure you have the following packages installed:
  - Numpy 
  - Scipy
  - Depending on which format you wish to use install:
    - Tensorflow
    - Keras
### Linux
- make sure you have installed TkInter: https://wiki.python.org/moin/TkInter
  - if you have anaconda run:
        ```conda install -c anaconda tk```
## Parsers available for neural networks created in the following libraries:
- Keras
## Verification Tools Currently Supported
- Reluplex
- Sherlock
## Translating into other model formats
To convert into the formats of other tools such as [Caffe2](https://caffe2.ai/docs/getting-started.html?platform=mac&configuration=prebuilt) [PyTorch] [Matlab](https://www.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format) or [several others](http://onnx.ai/getting-started) use the ONNX converter which can be found [here](https://github.com/onnx/tutorials)
## Contact
For questions please contact 
1. Diego Manzanas: diego.manzanas.lopez@vanderbilt.edu
2. Patrick Musau: patrick.musau@vanderbilt.edu

