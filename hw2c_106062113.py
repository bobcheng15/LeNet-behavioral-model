import torch 
import csv
import json
import numpy as np
import time
import numba as nb
import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt

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

    def inference(self, input_activation: np.array, bit_width: int): 
        '''
        Description:
            Function that carries out the inference of the layer
        Paremeter(s):
            input_activation := the input activation of this layer, an np array
            bit_width        := the bit width of the partial sum
        Return Value(s):
            output_activation := the output activation of this layer, an np array
        Exception(s):
            N/A
        '''
        self.bias_weight = np.clip(self.bias_weight, a_min=-1 * 2 ** bit_width, a_max=2 ** bit_width - 1)
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
        self.FullConCompute(input_activation, partial_sum, self.weight, self.is_biased, self.bias_weight, self.num_input_channel, self.activation_type, bit_width)
        # quantize the output activation by scaling it.
        output_activation = self.activation_scale * partial_sum
        # round the output activation and clip the activations that is out of range.
        output_activation = np.clip(output_activation, a_min=-128, a_max=127).round()
        # convert the type of the output activation 
        output_activation = output_activation.astype(np.int8)
        return output_activation, output_collection

    @staticmethod
    @nb.jit()      
    def FullConCompute(input_activation, partial_sum, weight, is_biased, bias_weight, num_input_channel, activation_type, bit_width):
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
            bit_width(int)            : the bit with of the partial sum 
        Return Value(s):
            N/A
        '''
        for i in range(0, partial_sum.shape[0]):
            for j in range(0, partial_sum.shape[1]):
                    for l in range(0, num_input_channel):
                        partial_sum[i][j] += input_activation[i][l] * weight[j][l]
                        if partial_sum[i][j] > 2 ** bit_width - 1: 
                            partial_sum[i][j] = 2 ** bit_width - 1
                            print("OVERFLOW +")
                        elif partial_sum[i][j] < -1 * 2 ** bit_width:
                            # print(partial_sum[i][j])
                            partial_sum[i][j] = -1 * 2 ** bit_width
                            print("LL OVERFLOW -")
                    # add the bias
                    partial_sum[i][j] += bias_weight[j]
                    if partial_sum[i][j] > 2 ** bit_width - 1: 
                            partial_sum[i][j] = 2 ** bit_width - 1
                            print("OVERFLOW +")
                    elif partial_sum[i][j] < -1 * 2 ** bit_width:
                        # print(partial_sum[i][j])
                        partial_sum[i][j] = -1 * 2 ** bit_width
                        print("LL OVERFLOW -")
                    # applies activation function, if this layer have one
                    partial_sum[i][j] = 0 if activation_type == 'ReLU' and partial_sum[i][j] < 0 else partial_sum[i][j]
                 
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
        
    def inference(self, input_activation: np.array, bit_width: int): 
        '''
        Description:
            Function that carries out the inference of the layer
        Paremeter(s):
            input_activation := the input activation of this layer, an np array
            bit_width        := not used 
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
                                    if partial_sum[n][m][p][q] > 2 ** bit_width - 1: 
                                        partial_sum[n][m][p][q] = 2 ** bit_width - 1
                                        print("OVERFLOW +")
                                    elif partial_sum[n][m][p][q] < -1 * 2 ** bit_width:
                                        partial_sum[n][m][p][q] = -1 * 2 ** bit_width
                                        print("OVERFLOW -")
                        # relu activation    
                        partial_sum[n][m][p][q] = 0 if partial_sum[n][m][p][q] < 0 else partial_sum[n][m][p][q]
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
    def __init__(self, input_scale):
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
        self.layers = []
        self.output_collection = [np.zeros((10000, 4704)), np.empty(0), np.zeros((10000, 1600)), np.empty(0), np.zeros((10000, 120)), np.zeros((10000, 84)), np.zeros((10000, 10))]
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
    
    def inference(self, input_activation, num_batch, bit_width):
        '''
        Description:
            Function that performs inference of the network.
        Parameter(s):
            input_activation(np.arry): the original input image.
            num_batch(int)           : the id of the batch the input activation belongs to 
            bit_width(list)          : the bit width of the partial sum for each layer
        Return Value(S):
            output_activation(np.array): the final output of the network.
        Exception(s):
            N/A
        '''
        # normalize the input activation.
        # input_activation = self.normalize_input(input_activation)
        # scale the input with the input scale and clip it to np.int8
        quantized_input = input_activation * self.input_scale
        quantized_input = np.clip(quantized_input, a_min=-128, a_max=127).round()
        quantized_input = quantized_input.astype(np.int8)
        # Feed teh quantized input to the rest of the network, layer by layer
        output_activation = quantized_input
        count = 0
        for layer in self.layers:
            output_activation, collection = layer.inference(output_activation, bit_width[count])
            self.output_collection[count][num_batch * 4: num_batch * 4 + output_activation.shape[0]] = collection
            count += 1
        return output_activation                        

def read_weight(weight_array, weight_file):
    '''
    Descritption:
        Function that reads the weight of a layer from a given csv file and store
        then into degignated np.array
    Parameter(s):
        weight_array(np.array): the np.array to store the weight read from the csv file
        weight_file(str)      : the path to the csv file that store the weight of the layer
    Return Value(s):
        None
    '''
    with open(weight_file) as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        # if it is a convolutional layer
        if (len(weight_array.shape) == 4):
            # read the content of the csv file and store them in the 
            for i in range(0, weight_array.shape[0]):
                for j in range(0, weight_array.shape[1]):
                    for k in range(0, weight_array.shape[2]):
                        for l in range(0, weight_array.shape[3]):
                            # note that the scientific notation have to first be converted to float
                            # then integer.
                            weight_array[i][j][k][l] = int(float(next(csv_reader)[0]))
        #weight of a linear layer
        elif(len(weight_array.shape) == 2):
            for i in range(0, weight_array.shape[0]):
                for j in range(0, weight_array.shape[1]):
                    weight_array[i][j] = int(float(next(csv_reader)[0]))
        # input activation
        elif(len(weight_array.shape) == 3):
            for i in range(0, weight_array.shape[0]):
                for j in range(0, weight_array.shape[1]):
                    for k in range(0, weight_array.shape[2]):
                         weight_array[i][j][k] = (float(next(csv_reader)[0]))
        #bias
        else:
            for i in range(0, weight_array.shape[0]):
                weight_array[i] = int(float(next(csv_reader)[0]))

                
def read_activation_scale(scale_file, layer_name):
    '''
    Description:
        Function that read the output activation quantization scale from a json file for a epecific layer
    Parameter(s):
        scale_file(str): path to the file that stores the activation scale for all the layers
        layer_name(str): the name of the layer whose activation scale is to be extracted
    Return Value(s):
        the output activation quantization scale of the layer specified in the parameter
    '''
    with open(scale_file) as fp:
        all_scale = json.load(fp)
        return all_scale[layer_name]




def create_network():
    # read input_activation
    input_scale = read_activation_scale('./parameters/scale.json', 'input_scale')
    # initialize network
    network = Network(input_scale)
    # conv1
    conv1_scale = read_activation_scale('./parameters/scale.json', 'conv1_output_scale')
    conv1_weight = np.zeros((6, 3, 5, 5), dtype=np.int8)
    read_weight(conv1_weight, './parameters/weights/conv1.weight.csv')
    network.add_layer(Conv2dLayer(3, 5, 6, conv1_weight, conv1_scale))
    # pool
    network.add_layer(MaxPoolLayer(6, 2, 6))
    # conv2
    conv2_scale = read_activation_scale('./parameters/scale.json', 'conv2_output_scale')
    conv2_weight = np.zeros((16, 6, 5, 5), dtype=np.int8)
    read_weight(conv2_weight, './parameters/weights/conv2.weight.csv')
    network.add_layer(Conv2dLayer(6, 5, 16, conv2_weight, conv2_scale))
    # pool
    network.add_layer(MaxPoolLayer(16, 2, 16))
    # fc1
    fc1_scale = read_activation_scale('./parameters/scale.json', 'fc1_output_scale')
    fc1_weight = np.zeros((120, 16 * 5 * 5), dtype=np.int8)
    read_weight(fc1_weight, './parameters/weights/fc1.weight.csv')
    network.add_layer(LinearLayer(16 * 5 * 5, 1, 120, fc1_weight, 'ReLU', fc1_scale, False, None))
    # fc2
    fc2_scale = read_activation_scale('./parameters/scale.json', 'fc2_output_scale')
    fc2_weight = np.zeros((84, 120), dtype=np.int8)
    read_weight(fc2_weight, './parameters/weights/fc2.weight.csv')
    network.add_layer(LinearLayer(120, 1, 84, fc2_weight, 'ReLU', fc2_scale, False, None))
    # fc3
    fc3_scale = read_activation_scale('./parameters/scale.json', 'fc3_output_scale')
    fc3_weight = np.zeros((10, 84), dtype=np.int8)
    read_weight(fc3_weight, './parameters/weights/fc3.weight.csv')
    fc3_bias = np.zeros((10), dtype=np.int32)
    read_weight(fc3_bias, './parameters/weights/fc3.bias.csv')
    network.add_layer(LinearLayer(84, 1, 10, fc3_weight, 'None', fc3_scale, True, fc3_bias))
    return network

if __name__ == "__main__":
    # intialize the pytorch dataset with data transform 
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=4)
    #create the network
    network = create_network()
    start = time.time()
    correct_count = 0
    total_count = 0
    # the bit width of the partial sums of each quantized layer
    # notice that this veriable is omitted in max pooling layers
    # so I set the bit width of these layers to 32
    bit_widths = [18, 32, 18, 32, 19, 18, 17]
    baseline_acc = 52.79
    
    for step, (input_activation, label) in enumerate(tqdm.tqdm(testloader)):
        # inference with the batch of data
        output_activation = network.inference(input_activation.cpu().numpy(), step, bit_widths)
        # take argmax on the activation output to obtain the prediction 
        output_label = np.argmax(output_activation, axis=1)
        # check the prediction against the ground truth 
        for i in range(4):
            total_count += 1
            if output_label[i] == label[i]:
                correct_count += 1
    # print the accuracy
    accuracy = correct_count/total_count * 100
    print("Accuracy: ", accuracy, "%")
    # test the difference between the original implementation
    # and the one with the current partial sum bit width
    fig, axs = plt.subplots(5)
    plt.tight_layout()
    plt.yscale('log')
    axs[0].title.set_text('conv1')
    axs[1].title.set_text('conv2')
    axs[2].title.set_text('fc1')
    axs[3].title.set_text('fc2')
    axs[4].title.set_text('fc3')
    axs[0].hist(network.output_collection[0].flatten())
    axs[1].hist(network.output_collection[2].flatten())
    axs[2].hist(network.output_collection[4].flatten())
    axs[3].hist(network.output_collection[5].flatten())
    axs[4].hist(network.output_collection[6].flatten())
    plt.savefig('c.png')
    print("conv1 output activation range: ", max(network.output_collection[0].flatten()),  min(network.output_collection[0].flatten()))
    print("conv2 output activation range: ", max(network.output_collection[2].flatten()), min(network.output_collection[2].flatten()))
    print("fc1 output activation range: ", max(network.output_collection[4].flatten()), min(network.output_collection[4].flatten()))
    print("fc1 output activation range: ", max(network.output_collection[5].flatten()), min(network.output_collection[5].flatten()))
    print("fc1 output activation range: ", max(network.output_collection[6].flatten()), min(network.output_collection[6].flatten()))

        
    
        



