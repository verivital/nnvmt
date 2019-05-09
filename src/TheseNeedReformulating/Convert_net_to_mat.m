%% Convert net (network object) to mat

% load network
file = 'controller_Lcontainer_3in.mat';
load(file);
net = netc;

% get NN attributes
W = {net.IW{1} net.LW{2} net.LW{6}};
b = {net.b{1} net.b{2} net.b{3}};
activation_fcns = ['relu  ';'relu  ';'linear'];
layer_sizes = [net.layers{1}.size net.layers{2}.size net.layers{3}.size];
number_of_inputs = size(net.IW{1},2);
number_of_outputs = size(W{end},1);
number_of_layers = length(b);
% save network
save(file,'W','b','layer_sizes','activation_fcns','number_of_inputs',...
    'number_of_outputs','number_of_layers');