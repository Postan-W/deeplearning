#用numpy编写一个RNN网络
import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps,input_features))
state_t = np.zeros((output_features,))

W = np.random.random((output_features,input_features))
U = np.random.random((output_features,output_features))
b = np.random.random((output_features,))

successive_output = []
for input_t in inputs:
    output_t = np.tanh(np.dot(W,input_t) + np.dot(U,state_t) + b)
    successive_output.append(output_t)
    state_t = output_t
print(successive_output[-1])