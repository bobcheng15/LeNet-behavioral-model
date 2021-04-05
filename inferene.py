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
    input_scale = Utils.read_activation_scale('./parameters/scale.json', 'input_scale')
    network = Network(input_scale)
    conv1_scale = Utils.read_activation_scale('./parameters/scale.json', 'conv1_output_scale')
    conv1_weight = np.zeros((6, 3, 5, 5), dtype=np.int8)
    Utils.read_weight(conv1_weight, './parameters/weights/conv1.weight.csv')
    network.add_layer(Conv2dLayer(3, 5, 6, conv1_weight, conv1_scale))
    network.add_layer(MaxPoolLayer(6, 2, 6))
    conv2_scale = Utils.read_activation_scale('./parameters/scale.json', 'conv2_output_scale')
    conv2_weight = np.zeros((16, 6, 5, 5), dtype=np.int8)
    Utils.read_weight(conv2_weight, './parameters/weights/conv2.weight.csv')
    network.add_layer(Conv2dLayer(6, 5, 16, conv2_weight, conv2_scale))
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
    network = create_network()
    # input_activation = np.zeros((3, 32, 32), dtype=float)
    # Utils.read_weight(input_activation, './parameters/activations/input.csv')
    # input_activation = np.expand_dims(input_activation, axis=0)
    start = time.time()
    # output_activation = network.inference(input_activation)
    count = 0
    total_count = 0
    for step, (input_activation, label) in enumerate(tqdm.tqdm(testloader)):
        output_activation = network.inference(input_activation.cpu().numpy(), step)
        output_label = np.argmax(output_activation, axis=1)
        for i in range(4):
            total_count += 1
            if output_label[i] == label[i]:
                count += 1
    print(count/total_count)
    end = time.time()
    print("Time taken: ", end - start)
    fig, axs = plt.subplots(5, gridspec_kw={'width_ratios': [5]}, figsize=(15,15))
    #adjust the layout of the subplots so the title of the plot don't overlap with each other
    plt.tight_layout()
    plt.yscale('log')
    #set the title of each subplot
    axs[0].title.set_text('conv1')
    
    print("conv1 max:", np.amax(network.output_collection[0]))
    print("conv1 min:", np.amin(network.output_collection[0]))
    print("conv2 max:", np.amax(network.output_collection[2]))
    print("conv2 min:", np.amin(network.output_collection[2]))
    print("fc1 max:", np.amax(network.output_collection[4]))
    print("fc1 min:", np.amin(network.output_collection[4]))
    print("fc2 max:", np.amax(network.output_collection[5]))
    print("fc2 min:", np.amin(network.output_collection[5]))
    print("fc3 max:", np.amax(network.output_collection[6]))
    print("fc3 min:", np.amin(network.output_collection[6]))
    axs[1].title.set_text('conv2')
    axs[2].title.set_text('fc1')
    axs[3].title.set_text('fc2')
    axs[4].title.set_text('fc3')
    #plot the histogram
    axs[0].hist(network.output_collection[0].flatten())
    axs[1].hist(network.output_collection[2].flatten())
    axs[2].hist(network.output_collection[4].flatten())
    axs[3].hist(network.output_collection[5].flatten())
    axs[4].hist(network.output_collection[6].flatten())
    
    plt.savefig('a.png')



