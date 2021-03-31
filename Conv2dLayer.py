import numpy as np
import Utils
from Layer import Layer
'''
Class: Conv2dLayer
    Description:
        The class inherits the 'Layer' class and implement the 2d convolutional layer in the network.
    Member Variable(s):
        weight(np.array)       : the weight of the layer, a np.int8 numpy array of dimension (M, C, R, S).
        activation_type(int)   : the type of activation functino used in the layer, a string that's either 'ReLU' or 'None'.
        activation_scale(float): the output activation quantization scale for quantizing the output activation
'''
class Conv2dLayer(Layer):
    def __init__(self, num_input_channel ,window_size, num_kernel, pretrained_weight, activation_type, activation_scale):
        '''
        Description:
            Inherits from the constructor of the base class, the contructor function of class 'Conv2dLayer'.
            This contructor set the dimension of the layer weight, 
            and store the pretrained weight for inference.
        Parameter(s):
            num_input_channel(int): the number of channel (C) of the input activation.
            window_size(int)           : convolution kernel window size (R, S).
            num_kernel(int)            : number of kernels in the layer (M).
            pretrained_weight(np.array): the pretrained weight of the layer, an numpy array of size (M, C, R, S)
            activation_type(str)       : the type of activation functino used in the layer, a string that's either 'ReLU' or 'None'.
            activation_scale(float)    : quantization scale used to quantize the output activation weight.
        Return Value(s):
            N/A
        Exception(s):
            N/A
        '''
        super().__init__(num_input_channel, window_size, num_input_channel)
        # create numpy array to store weight (R, S, C, M), and store the pretrained weight
        self.weight = np.zeros((num_kernel, num_input_channel ,window_size, window_size), dtype=np.int8)
        np.copyto(self.weight, pretrained_weight)
        # set the activation type
        self.activation_type = activation_type
        # set the activation quantization scale
        self.activation_scale = activation_scale
        
    def inference(self, input_activation: np.array): 
        '''
        Description:
            Function that carries out the inference of the layer
        Paremeter(s):
            input_activation := the input activation of this layer, an np array
        Return Value(s):
            output_activation := the output activation of this layer, an np array
        Exception(s):
            N/A
        '''
        # case input_activation to np.uint8 (just to make sure)
        input_activation = input_activation.astype(np.uint8)
        # create np.array to store the partial sum
        partial_sum = np.zeros((self.num_kernel, input_activation.shape[2] - self.window_size + 1,
                                input_activation.shape[2] - self.window_size + 1), dtype=np.int32)
        # accumulate the partial sum
        for i in range(0, partial_sum.shape[0]):
            for j in range(0, partial_sum.shape[1]):
                for k in range(0, partial_sum.shape[2]):
                    for l in range(0, self.num_input_channel):
                        for m in range(0, self.window_size):
                            for n in range(0, self.window_size):
                                partial_sum[i][j][k] += input_activation[l][j + m][n + n] * self.weight[i][l][m][n]
        # apply the activation function, if an the layer have one.
        if self.activation_type == 'ReLU':
            output_activation = np.clip(partial_sum, a_min=0, a_max=np.Inf)
        # quantize the output activation by scaling it.
        output_activation = self.activation_scale * output_activation
        # round the output activation and clip the activations that is out of range.
        output_activation = np.clip(output_activation, a_min=-128, a_max=127).round()
        # convert the type of the output activation 
        output_activation = output_activation.astype(np.int8)
        return output_activation


if __name__ == "__main__":
    # create numpy array for the weight of conv1
    conv1_weight = np.zeros((6, 3, 5, 5), dtype=np.int8)
    # read the weight and the scale of the first layer, and create the layer
    Utils.read_weight(conv1_weight, './parameters/weights/conv1.weight.csv')
    conv1_scale = Utils.read_activation_scale('./parameters/scale.json', 'conv1')
    conv1_layer = Conv2dLayer(3, 5, 6, conv1_weight, 'ReLU', conv1_scale)
    sample_input = np.zeros((3, 32, 32))
    sample_output = conv1_layer.inference(sample_input)
    print(sample_output)





                


        


        
    

