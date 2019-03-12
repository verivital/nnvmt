%% Convert neural networks saved as struct to net objects 
% clc;clear
%%  Load the neural network file
%file = 'SingleCarPlant'; %Has to be a .mat file
% load('SingleCarPlant'); %all variables from file are loaded into workspace
% file = SingleCarPlant; 

%% Parameters of the network
nl = double(file.number_of_layers); % # of layers
ni = double(file.number_of_inputs); % # of inputs
no = double(file.number_of_outputs); % # of outputs

%% Transform names of activations function to its corresponding name in MATLAB
act = erase(string(file.activation_fcns)," "); %transform to strings

for i = 1:nl
    if act(i) == "relu" %ReLU activation function
        lystype{i} = 'poslin'; 
    elseif act(i) == "linear" %Linear Activation function
        lystype{i} = 'purelin';
    elseif act(i) == "sigmoid" %Sigmoid activation function
        lystype{i} = 'logsig';
    elseif act(i) == "tanh" %Hyperbolic tangent activation function
        lystype{i} = 'tansig';
    elseif act(i) == "relu1" %Saturation linear function from 0 to 1
        lystype{i} = 'satlin'
    elseif act(i) == "relu2" %Saturation linear function from -1 to 1
        lystype{i} = 'satlins'
    else
        disp("The activation function of layer "+i+" is currently not supported");
    end
end   
% for i = 1:nl
%     if contains(string(file.activation_fcns(i,:)),"relu")
%         lystype{i} = 'poslin';
%     elseif contains(string(file.activation_fcns(i,:)),"linear")
%         lystype{i} = 'purelin';
%     elseif contains(string(file.activation_fcns(i,:)),"sigmoid")
%         lystype{i} = 'logsig';
%     elseif contains(string(file.activation_fcns(i,:)),"tanh")
%         lystype{i} = 'tansig';
%     %elseif 
%     else
%         lystype{i} = file.activation_fcns(i,:);
%     end
% end   

    %% Create and define feedforward network
net = feedforwardnet(double(file.layer_sizes(1:(end-1))));
% net.inputs{1}.size = ni;
net.inputs{1}.processFcns = {}; %Remove preprocessing functions in the inputs and outputs
net.outputs{nl}.processFcns = {};

%% Transfer functions
for i = 1:nl
    net.layers{i}.size = length(file.b{1,i});
    %net.layers{i}.transferFcn = 'poslin'; %poslin = relu
    net.layers{i}.transferFcn = lystype{i}; %poslin = relu
    length(file.b{1,i});
end

%% Weights matrics
net.inputs{1}.size = ni;
for i = 1:nl-1
    net.LW{i+1,i} = double(file.W{i+1});
end
net.IW{1,1} = double(file.W{1});

%% Bias matrices
for i =1:nl
    net.b{i} = double(file.b{i});
end

%% Save files
%savefile = 'ControllerCartPole.mat';
%save(savefile,'net');

%% Generate simulink file
% gensim(net)

%% Future work
% Need to fix some errors either here or in the keras parser, since Feiyang
% built a keras model where he has different layers for the activation, so
% it works similar to tensorflow (linear -> relu -> linear -> relu ->
% linear