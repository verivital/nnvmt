import numpy as np
np.set_printoptions(threshold=np.inf,linewidth=100000)
import scipy.io as sio

mat_contents = sio.loadmat('/home/musaup/Documents/Research/Experiments/MNISTModelsConverted/2000_10output_model.mat', squeeze_me=True)
del mat_contents['__header__']
del mat_contents['__globals__']
del mat_contents['__version__']


#get the weights and biases from the .mat file
weights=mat_contents['W']
biases=mat_contents['b']
print(weights,biases)