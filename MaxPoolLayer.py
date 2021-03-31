import numpy as np
import Utils
from Layer import Layer
'''
Class: MaxPoolLayer
    Description:
        The class inherits the 'Layer' class and implement the max pooling layer in the network.
    Member Variable(s):
        N/A
'''
class LinearLayer:
    def __init__(self, num_input_channel ,window_size, num_kernel, pretrained_weight, activation_type, activation_scale, bias, biase_weight):
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
        input_activation = input_activation.astype(np.uint8)
        # create np.array to store the partial sum
        output_activation = np.array((self.num_kernel, input_activation.shape[3] - self.window_size + 2,
                                input_activation.shape[3] - self.window_size + 2), dtype=np.int32)
        # max pooling operation

        # apply the activation function, if an the layer have one.
        if activation_type == 'ReLU':
            output_activation = np.clip(partial_sum, a_min=0, a_max=np.Inf)
        elif activation == 'None':
            pass
        else:
            raise NotImplementedError
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
    conv1_layer = LinearLayer(3, 5, 6, conv1_weight, 'None', conv1_scale, False, None)
    print(conv1_layer)