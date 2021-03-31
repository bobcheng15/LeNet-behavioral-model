import Layer
import numpy as np
'''
Class: Network
    Description:
        The class for a quantized neural network containing arbitary number of layers.
    Member Variable(s):
        num_layers(int)                  : the number of layers in the network.
        input_scale(flaot)               : the input activation quantization scale.
        layers([Layer])                  : the list that stores all the layers.
        input_mean((float, float, float)): the mean of the input activation.
        input_std((flaot, float, float)) : the standard deviation of the input image.
'''

class Network:
    def __init__(self, input_scale, input_mean, input_std):
        '''
        Description:
            The constructor of the Network class. It inistialize the number of layers, the list of layers, and 
            store the input activation quantization scale for inference.
        Parameter(s):
            input_scale(float)               : the input activation quantization scale.
            input_mean((float, float, float)): the mean of the input activation.
            input_std((float, float, float)) : the standard deviation of the input activation.
        Return Value(s): 
            N/A
        Exception(s):
            N/A
        '''
        # initialize number of layer, layer list, input scale
        self.num_layers = 0;
        self.input_scale = input_scale
        self.input_mean = input_mean
        self.input_std = input_std
        self.layers = []
        
    
    def normalize_input(self, input_activation):
        '''
        Description:
            This function normalize the input activation. It is done by first subtracting the mean from the activation, 
            then divide it with the standard deviation.
        Parameter(s):
            input_activation(np.array): the input activation.
        Return Value(s):
            input_activation(np.array): the normalized input activation.
        Exception(s);
            N/A
        '''
        input_activation = (input_activation - self.input_mean) / self.input_std
        return input_activation

    def add_layer(self, new_layer: Layer):
        '''
        Description:
            Function that appends layer to the network.
        Parameter(s):
            new_layer(Layer): layer to be appended.
        Return Value(s):
            N/A
        Exception(s):
            N/A
        '''
        self.layers.append(new_layer)
        self.num_layers += 1
    
    def inference(self, input_activation):
        '''
        Description:
            Function that performs inference of the network.
        Parameter(s):
            input_activation(np.arry): the original input image.
        Return Value(S):
            output_activation(np.array): the final output of the network.
        Exception(s):
            N/A
        '''
        # normalize the input activation.
        input_activation = self.normalize_input(input_activation)
        # scale the input with the input scale and clip it to np.int8
        quantized_input = input_activation * self.input_scale
        quantized_input = np.clip(quantized_input, a_min=-128, a_max=127).round()
        quantized_input = quantized_input.astype(np.int8)
        # Feed teh quantized input to the rest of the network, layer by layer
        output_activation = quantized_input
        for layer in self.layers:
            output_activation = layer.inference(output_activation)
        return output_activation
        
        
        

        