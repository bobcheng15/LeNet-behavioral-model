import numpy as np
import Utils
from Layer import Layer
import numba as nb
'''
Class: MaxPoolLayer
    Description:
        The class inherits the 'Layer' class and implement the max pooling layer in the network.
    Member Variable(s):
        N/A
'''
class MaxPoolLayer(Layer):
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
        super().__init__(num_input_channel, window_size, num_input_channel)
        
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
        input_activation = input_activation.astype(np.int8)
        # create np.array to store the partial sum
        output_activation = np.zeros((input_activation.shape[0], input_activation.shape[1], int(input_activation.shape[2] / self.window_size), int(input_activation.shape[3] / self.window_size)), dtype=np.int32)
        # max pooling operation, once again with a static method
        self.MaxPool(input_activation, output_activation, self.window_size)
        # round the output activation and clip the activations that is out of range.
        output_activation = np.clip(output_activation, a_min=-128, a_max=127).round()
        # convert the type of the output activation 
        output_activation = output_activation.astype(np.int8)
        return output_activation, np.empty(0)
    @staticmethod    
    @nb.jit()
    def MaxPool(input_activation, output_activation, window_size):
        '''
        Description:
            Function that carry out the computation of the max pooling layer 
        Parameter(s):
            input_activation(np.array) : the input activation to this layer
            output_activation(np.array): the down sampled output activation
            weight(np.array)           : the weight of this layer
            window_size(int)           : the size of the down sample kernel
        '''
        for i in range(output_activation.shape[0]):
            for j in range(output_activation.shape[1]):
                for k in range(output_activation.shape[2]):
                    for l in range(output_activation.shape[3]):
                        MAX = np.NINF
                        for w in range(window_size):
                            for h in range(window_size):
                                if (input_activation[i][j][k * window_size + w][l * window_size + h] > MAX):
                                    MAX = input_activation[i][j][k * window_size + w][l * window_size + h]
                        output_activation[i][j][k][l] = MAX


if __name__ == "__main__":
    maxpool_layer = MaxPoolLayer(2, 2, 2)
    input_activation =  np.array([[[[1, 2, 3, 4], [1, 2, 3, 4]], [[4, 4, 3, 4], [1, 2, 3, 4]]]])
    print(input_activation)
    print(input_activation.shape)
    output_activation = maxpool_layer.inference(input_activation)
    print(output_activation)