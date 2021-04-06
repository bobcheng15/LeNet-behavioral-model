import numpy as np
import Utils
from Layer import Layer
import numba as nb

'''
Class: Conv2dLayer
    Description:
        The class inherits the 'Layer' class and implement the 2d convolutional layer in the network.
    Member Variable(s):
        weight(np.array)       : the weight of the layer, a np.int8 numpy array of dimension (M, C, R, S).
        activation_scale(float): the output activation quantization scale for quantizing the output activation
'''
class Conv2dLayer(Layer):
    def __init__(self, num_input_channel ,window_size, num_kernel, pretrained_weight, activation_scale):
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
            activation_scale(float)    : quantization scale used to quantize the output activation weight.
        Return Value(s):
            N/A
        Exception(s):
            N/A
        '''
        super().__init__(num_input_channel, window_size, num_kernel)
        # create numpy array to store weight (R, S, C, M), and store the pretrained weight
        self.weight = np.zeros((num_kernel, num_input_channel ,window_size, window_size), dtype=np.int8)
        np.copyto(self.weight, pretrained_weight)
        # set the activation quantization scale
        self.activation_scale = activation_scale

    def inference(self, input_activation: np.array, bit_width: int): 
        '''
        Description:
            Function that carries out the inference of the layer
        Paremeter(s):
            input_activation(np.array): the input activation of this layer, an np array
            bit_width(int)            : the bit width of the partial sum
        Return Value(s):
            output_activation(np.array): the output activation of this layer, an np array
            output_collection(np.array): the flattened unquantized output activation
        Exception(s):
            N/A
        '''
        # case input_activation to np.uint8 (just to make sure)
        input_activation = input_activation.astype(np.int8)
        # create np.array to store the partial sum
        partial_sum = np.zeros((input_activation.shape[0], self.num_kernel, input_activation.shape[2] - self.window_size + 1, input_activation.shape[2] - self.window_size + 1), dtype=np.int32)
        # accumulate the partial sum
        # this method is implemented using a static method
        # to avoid dealing with jitclass
        self.convolve(input_activation, partial_sum, self.weight, self.window_size, self.num_input_channel, bit_width) 
        # collect the unquantized partial sum
        output_collection = partial_sum.reshape(partial_sum.shape[0], -1)        
        # quantize the output activation by scaling it.
        output_activation = self.activation_scale * partial_sum
        # round the output activation and clip the activations that is out of range.
        output_activation = np.clip(output_activation, a_min=-128, a_max=127).round()
        # convert the type of the output activation 
        output_activation = output_activation.astype(np.int8)
        
        return output_activation, output_collection
    @staticmethod
    @nb.jit()
    def convolve(input_activation, partial_sum, weight, window_size, num_input_channel, bit_width):
        '''
        Description:
            Function that carries out the convolution process
        Parameter(s):
            input_activation(np.array): the input activation to the convolution
            partial_sum(np.array)     : the partial sum to be accumulated during convolution
            weight(np.array)          : the weight of the kernel
            window_size(int)          : the windows size of the kernel
            num_input_channel(int)    : number of channals 'input_activation' contains
            bit_width(int)            : the bit width of the partial sum
        Return Value(s)
            N/A
        '''
        for n in range(0, partial_sum.shape[0]):
            for m in range(0, partial_sum.shape[1]):
                for p in range(0, partial_sum.shape[2]):
                    for q in range(0, partial_sum.shape[3]):
                        for r in range(0, window_size):
                            for s in range(0, window_size):
                                for c in range(0, num_input_channel):
                                    h = p + r
                                    w = q + s
                                    partial_sum[n][m][p][q] += input_activation[n][c][h][w] * weight[m][c][r][s] 
                        # relu activation    
                        partial_sum[n][m][p][q] = 0 if partial_sum[n][m][p][q] < 0 else partial_sum[n][m][p][q]
                        # reduce bit width to 19 bits
                        if partial_sum[n][m][p][q] > 2 ** bit_width - 1: 
                            partial_sum[n][m][p][q] = 2 ** bit_width - 1
                            # print("OVERFLOW +")
                        elif partial_sum[n][m][p][q] < -1 * 2 ** bit_width:
                            partial_sum[n][m][p][q] = -1 * 2 ** bit_width
                            # print("OVERFLOW -")

   
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





                


        


        
    

