import torch 
import Utils
import numpy as np
from Network import Network
from Conv2dLayer import Conv2dLayer
from MaxPoolLayer import MaxPoolLayer
from LinearLayer import LinearLayer
import time
import numba as nb
import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt

def create_network():
    # read input_activation
    input_scale = Utils.read_activation_scale('./parameters/scale.json', 'input_scale')
    # initialize network
    network = Network(input_scale)
    # conv1
    conv1_scale = Utils.read_activation_scale('./parameters/scale.json', 'conv1_output_scale')
    conv1_weight = np.zeros((6, 3, 5, 5), dtype=np.int8)
    Utils.read_weight(conv1_weight, './parameters/weights/conv1.weight.csv')
    network.add_layer(Conv2dLayer(3, 5, 6, conv1_weight, conv1_scale))
    # pool
    network.add_layer(MaxPoolLayer(6, 2, 6))
    # conv2
    conv2_scale = Utils.read_activation_scale('./parameters/scale.json', 'conv2_output_scale')
    conv2_weight = np.zeros((16, 6, 5, 5), dtype=np.int8)
    Utils.read_weight(conv2_weight, './parameters/weights/conv2.weight.csv')
    network.add_layer(Conv2dLayer(6, 5, 16, conv2_weight, conv2_scale))
    # pool
    network.add_layer(MaxPoolLayer(16, 2, 16))
    # fc1
    fc1_scale = Utils.read_activation_scale('./parameters/scale.json', 'fc1_output_scale')
    fc1_weight = np.zeros((120, 16 * 5 * 5), dtype=np.int8)
    Utils.read_weight(fc1_weight, './parameters/weights/fc1.weight.csv')
    network.add_layer(LinearLayer(16 * 5 * 5, 1, 120, fc1_weight, 'ReLU', fc1_scale, False, None))
    # fc2
    fc2_scale = Utils.read_activation_scale('./parameters/scale.json', 'fc2_output_scale')
    fc2_weight = np.zeros((84, 120), dtype=np.int8)
    Utils.read_weight(fc2_weight, './parameters/weights/fc2.weight.csv')
    network.add_layer(LinearLayer(120, 1, 84, fc2_weight, 'ReLU', fc2_scale, False, None))
    # fc3
    fc3_scale = Utils.read_activation_scale('./parameters/scale.json', 'fc3_output_scale')
    fc3_weight = np.zeros((10, 84), dtype=np.int8)
    Utils.read_weight(fc3_weight, './parameters/weights/fc3.weight.csv')
    fc3_bias = np.zeros((10), dtype=np.int32)
    Utils.read_weight(fc3_bias, './parameters/weights/fc3.bias.csv')
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
    bit_width = 19
    baseline_acc = 52.79
    while True:
        # iterate through the test set
        print("Bit-width: ", bit_width)
        for step, (input_activation, label) in enumerate(tqdm.tqdm(testloader)):
            # inference with the batch of data
            output_activation = network.inference(input_activation.cpu().numpy(), step, bit_width)
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
        if baseline_acc -  accuracy >= 1:
            break;
        else:
            correct_count = 0
            total_count = 0
            bit_width -= 1
    print("Minimum bit width: ", bit_width + 1)
        
       
   
    



