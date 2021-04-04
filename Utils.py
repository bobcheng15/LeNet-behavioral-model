import csv 
import json
import numba as nb
import numpy as np


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

@nb.jit()      
def FullConCompute(input_activation, partial_sum, weight, is_biased, bias_weight, num_input_channel):
    for i in range(0, partial_sum.shape[0]):
        for j in range(0, partial_sum.shape[1]):
                for l in range(0, num_input_channel):
                    product = np.zeros((1), dtype=np.int16)
                    product = input_activation[i][l] * weight[j][l]
                    partial_sum[i][j] += product
        # add the bias to the partial sum
        if is_biased:
            partial_sum[i] += bias_weight      

@nb.jit()
def convolve(input_activation, partial_sum, weight, window_size, num_input_channel):
    '''
    Description:
        Function that carries out the convolution process
    Parameter(s):
        input_activation(np.array): the input activation to the convolution
        partial_sum(np.array)     : the partial sum to be accumulated during convolution
        weight(np.array)          : the weight of the kernel
        window_size(int)          : the windows size of the kernel
        num_input_channel(int)    : number of channals 'input_activation' contains
    Return Value(s)
        N/A
    '''
    for n in range(0, partial_sum.shape[0]):
        for m in range(0, partial_sum.shape[1]):
            for p in range(0, partial_sum.shape[2]):
                for q in range(0, partial_sum.shape[3]):
                    partial_sum[n][m][p][q] 
                    for r in range(0, window_size):
                        for s in range(0, window_size):
                            for c in range(0, num_input_channel):
                                h = p + r
                                w = q + s
                                product = np.zeros((1), dtype=np.int16)
                                product = input_activation[n][c][h][w] * weight[m][c][r][s]
                                partial_sum[n][m][p][q] += product      


def MaxPool(input_activation, output_activation, window_size):
    for i in range(output_activation.shape[0]):
        for j in range(output_activation.shape[1]):
            for k in range(output_activation.shape[2]):
                for l in range(output_activation.shape[3]):
                    # array used to collect the numbers in the max pooling window 
                    MAX = np.NINF
                    for w in range(window_size):
                        for h in range(window_size):
                            if (input_activation[i][j][k * window_size + w][l * window_size + h] > MAX):
                                output_activation[i][j][k][l] =  input_activation[i][j][k * window_size + w][l * window_size + h]
                                MAX = input_activation[i][j][k * window_size + w][l * window_size + h]