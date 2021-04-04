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



