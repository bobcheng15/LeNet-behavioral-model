import numpy as np
import Utils
'''
Class: Layer
    Description:
        The base class for all the layers in the neural network 
    Member Variable(s):
        num_input_channel(int): the number of channel (C) of the input activation.
        window_size(int)       :convolution kernel window size (R, S).
        num_kernel(int)        : number of kernels in the layer (M).
'''
class Layer:
    def __init__(self, num_input_channel ,window_size, num_kernel):
        '''
        Description:
            The contructor function of class 'Layer'. This contructor set the dimension of the layer weight, 
            and store the pretrained weight for inference.
        Parameter(s):
            num_input_channel(int): the number of channel (C) of the input activation.
            window_size(int)           : convolution kernel window size (R, S).
            num_kernel(int)            : number of kernels in the layer (M).
        Return Value(s):
            N/A
        Exception(s):
            N/A
        '''

        # set the convolution window size (R, S)
        self.window_size = window_size
        # set the number of convolutional kernel (M)
        self.num_kernel  = num_kernel
        # set the number of input channel (C)
        self.num_input_channel = num_input_channel
    
    def inference(self): 
        '''
        Description:
            Function that carries out the inference of the layer, override this 
            template method in the child class
        Paremeter(s):
            N/A
        Return Value(s):
            N/A
        Exception(s):
            NotImplementedError
        '''
        # don't call this method in the bsae class directly
        raise NotImplementedError



                


        


        
    

