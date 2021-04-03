import torch 
import Utils
import numpy as np
from Network import Network
from Conv2dLayer import Conv2dLayer
from MaxPoolLayer import MaxPoolLayer
from LinearLayer import LinearLayer

def create_network():
    input_scale = Utils.read_activation_scale('./parameters/scale.json', 'input_scale')
    network = Network(input_scale)
    conv1_scale = Utils.read_activation_scale('./parameters/scale.json', 'conv1_output_scale')
    conv1_weight = np.zeros((6, 3, 5, 5), dtype=np.int8)
    Utils.read_weight(conv1_weight, './parameters/weights/conv1.weight.csv')
    network.add_layer(Conv2dLayer(3, 5, 6, conv1_weight, 'ReLU', conv1_scale))
    network.add_layer(MaxPoolLayer(6, 2, 6))
    conv2_scale = Utils.read_activation_scale('./parameters/scale.json', 'conv2_output_scale')
    conv2_weight = np.zeros((16, 6, 5, 5), dtype=np.int8)
    Utils.read_weight(conv2_weight, './parameters/weights/conv2.weight.csv')
    network.add_layer(Conv2dLayer(6, 5, 16, conv2_weight, 'ReLU', conv2_scale))
    network.add_layer(MaxPoolLayer(16, 2, 16))
    fc1_scale = Utils.read_activation_scale('./parameters/scale.json', 'fc1_output_scale')
    fc1_weight = np.zeros((120, 16 * 5 * 5), dtype=np.int8)
    Utils.read_weight(fc1_weight, './parameters/weights/fc1.weight.csv')
    network.add_layer(LinearLayer(16 * 5 * 5, 1, 120, fc1_weight, 'ReLU', fc1_scale, False, None))
    fc2_scale = Utils.read_activation_scale('./parameters/scale.json', 'fc2_output_scale')
    fc2_weight = np.zeros((84, 120), dtype=np.int8)
    Utils.read_weight(fc2_weight, './parameters/weights/fc2.weight.csv')
    network.add_layer(LinearLayer(120, 1, 84, fc2_weight, 'ReLU', fc2_scale, False, None))
    fc3_scale = Utils.read_activation_scale('./parameters/scale.json', 'fc3_output_scale')
    fc3_weight = np.zeros((10, 84), dtype=np.int8)
    Utils.read_weight(fc3_weight, './parameters/weights/fc3.weight.csv')
    fc3_bias = np.zeros((10), dtype=np.int32)
    Utils.read_weight(fc3_bias, './parameters/weights/fc3.bias.csv')
    network.add_layer(LinearLayer(84, 1, 10, fc3_weight, 'None', fc3_scale, True, fc3_bias))
    return network

if __name__ == "__main__":
    network = create_network()
    input_activation = np.zeros((3, 32, 32), dtype=float)
    Utils.read_weight(input_activation, './parameters/activations/input.csv')
    input_activation = np.expand_dims(input_activation, axis=0)
    print(input_activation)
    output_activation = network.inference(input_activation)
    print(output_activation)

