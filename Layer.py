import numpy as np
'''
Class: Layer
    Description:
        The base class of all the layers in the neural network 
        Inherit this class to implement more specific layers
    Member Variable(s):
        num_input_channel := the number of channel (C) of the input activation
        window_size       := convolution kernel window size (R, S)
        num_kernel        := number of kernels in the layer (M)
        weight            := the weight of the layer, a np.int8 numpy array of dimension (M, C, R, S)
        activation_type   := the type of activation functino used in the layer, a string that's either 'ReLU' or 'None'.
        bias              := an indicator showing that whether this layer is biased or not
        bias_weight       := the weight of the bias in the lyaer, a np.int32 array
'''
class Layer:
    def __init__(self, num_input_channel ,window_size, num_kernel, pretrained_weight, activation_type, activation_scale, biase, biase_weight):
        '''
        Description:
            The contructor function of class 'Layer'. This contructor set the dimension of the layer weight, 
            and store the pretrained weight for inference.
        Parameter(s):
            num_input_channel := the number of channel (C) of the input activation.
            window_size       := convolution kernel window size (R, S).
            num_kernel        := number of kernels in the layer (M).
            pretrained_weight := the pretrained weight of the layer, an numpy array of size (M, C, R, S)
            activation_type   := the type of activation functino used in the layer, a string that's either 'ReLU' or 'None'.
            activation_scale  := quantization scale used to quantize the output activation weight.
            bias              := whether this layer is biased or not.
            bias_weight       := the weight of the bias.
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
        # create numpy array to store weight (R, S, C, M), and store the pretrained weight
        self.weight = np.zeros((num_kernel, num_input_channel ,window_size, window_size), dtype=np.int8)
        np.copyto(self.weight, pretrained_weight)
        # set the activation type
        self.activation_type = activation_type
        # set the activation quantization scale
        self.activation_scale = activation_scale
        # set the bias indicator, if biased, create np array and store the biase weight 
        self.is_biased = bias
        if is_biased:
            self.bias_weight = np.zeros((num_input_channel, window_size, window_size), dtype=np.int32)
            np.copyto(self.bias_weight, bias_weight)

    def inference(self, input_activation: np.array) -> np.array:
        '''
        Description:
            Function that carries our the inference of the layer
        Paremeter(s):
            input_activation := the input activation of this layer, an np array
        Return Value(s):
            output_activation := the output activation of this layer, an np array
        Exception(s):
            N/A
        '''
        input_activation = input_activation.astype(np.uint8)

        


        
    

