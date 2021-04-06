import numpy as np
import Utils
from Layer import Layer
import numba as nb
'''
Class: LinearLayer
    Description:
        The class inherits the 'Layer' class and implement the fully connected layer in the network.
    Member Variable(s):
        weight(int)            : the weight of the layer, a np.int8 numpy array of dimension (M, C, R, S)
        activation_type(int)   : the type of activation functino used in the layer, a string that's either 'ReLU' or 'None'.
        activation_scale(float): the output activation quantization scale for quantizing the output activation.
        bias(int)              : an indicator showing that whether this layer is biased or not
        bias_weight(int)       : the weight of the bias in the lyaer, a np.int32 array
'''
class LinearLayer(Layer):
    def __init__(self, num_input_channel ,window_size, num_kernel, pretrained_weight, activation_type, activation_scale, bias, bias_weight):
        '''
        Description:
            The contructor function of class 'Layer'. This contructor set the dimension of the layer weight, 
            and store the pretrained weight for inference.
        Parameter(s):
            num_input_channel(int): the number of channel (C) of the input activation.
            window_size(int)           : convolution kernel window size (R, S).
            num_kernel(int)            : number of kernels in the layer (M).
            pretrained_weight(np.array): the pretrained weight of the layer, an numpy array of size (M, C, R, S)
            activation_type(str)       : the type of activation functino used in the layer, a string that's either 'ReLU' or 'None'.
            activation_scale(float)    : quantization scale used to quantize the output activation weight.
            bias(bool)                 : whether this layer is biased or not.
            bias_weight(np.array)      : the weight of the bias.
        Return Value(s):
            N/A
        Exception(s):
            N/A
        '''
        super().__init__(num_input_channel, window_size, num_kernel)
        # create numpy array to store weight (R, S, C, M), and store the pretrained weight
        self.weight = np.zeros((num_kernel, num_input_channel), dtype=np.int8)
        np.copyto(self.weight, pretrained_weight)
        # set the activation type
        self.activation_type = activation_type
        # set the activation quantization scale
        self.activation_scale = activation_scale
        # set the bias indicator, if biased, create np array and store the biase weight 
        self.is_biased = bias
        self.bias_weight = np.zeros((num_kernel), dtype=np.int32)
        if self.is_biased:
            np.copyto(self.bias_weight, bias_weight)

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
        if len(input_activation.shape) != 2:
            input_activation = input_activation.astype(np.int8)
            input_activation_flatten = []
            for i in range(0, input_activation.shape[0]):
                temp = input_activation[i, :, :, :].flatten()
                input_activation_flatten.append(temp)
            input_activation = np.array(input_activation_flatten, dtype=np.int8)
        # create np.array to store the partial sum
        partial_sum = np.zeros((input_activation.shape[0], self.num_kernel), dtype=np.int32)
        # collect the unquantized partial sum
        output_collection = partial_sum.reshape(partial_sum.shape[0], -1)
        # use a static method to accumulate the partial sum since numba's class support is shit
        self.FullConCompute(input_activation, partial_sum, self.weight, self.is_biased, self.bias_weight, self.num_input_channel, self.activation_type)
        # quantize the output activation by scaling it.
        output_activation = self.activation_scale * partial_sum
        # round the output activation and clip the activations that is out of range.
        output_activation = np.clip(output_activation, a_min=-128, a_max=127).round()
        # convert the type of the output activation 
        output_activation = output_activation.astype(np.int8)
        return output_activation, output_collection

    @staticmethod
    @nb.jit()      
    def FullConCompute(input_activation, partial_sum, weight, is_biased, bias_weight, num_input_channel, activation_type):
        '''
        Description:
            Function that carry out the computation of the fully connected layer and applies the activation function
        Parameter(s):
            input_activation(np.array): the input activation to this layer
            partial_sum(np.array)     : the partial sum to be accumulated
            weight(np.array)          : the weight of this layer
            is_biased(bool)           : whether this layer is biased or not
            biase_weight(np.arry)     : the bias of this layer
            num_input_channel(int)    : the number of channel input_activation possesses
            activation_type(str)      : the type of the activation function, 'ReLU', or 'None'
        '''
        for i in range(0, partial_sum.shape[0]):
            for j in range(0, partial_sum.shape[1]):
                    for l in range(0, num_input_channel):
                        partial_sum[i][j] += input_activation[i][l] * weight[j][l]
                        if partial_sum[i][j] > 524287: 
                            partial_sum[i][j] = 524287
                            print("OVERFLOW +")
                        elif partial_sum[i][j] < -524288:
                            print(partial_sum[i][j])
                            partial_sum[i][j] = -524288
                            print("LL OVERFLOW -")
                    partial_sum[i][j] += bias_weight[j]
                    # reduce bit width to 19 bits.
                    if partial_sum[i][j] > 524287: 
                        partial_sum[i][j] = 524287
                        print("OVERFLOW +")
                    elif partial_sum[i][j] < -524288:
                        print(partial_sum[i][j])
                        partial_sum[i][j] = -524288
                        print("LL OVERFLOW -")
                    # applies activation functino, if this layer have one
                    partial_sum[i][j] = 0 if activation_type == 'ReLU' and partial_sum[i][j] < 0 else partial_sum[i][j]
                  

if __name__ == "__main__":
    # create numpy array for the weight of conv1
    conv1_weight = np.zeros((6, 3, 5, 5), dtype=np.int8)
    # read the weight and the scale of the first layer, and create the layer
    Utils.read_weight(conv1_weight, './parameters/weights/conv1.weight.csv')
    conv1_scale = Utils.read_activation_scale('./parameters/scale.json', 'conv1')
    conv1_layer = LinearLayer(3, 5, 6, conv1_weight, 'None', conv1_scale, False, None)
    print(conv1_layer)