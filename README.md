<!-- ![Teaism](logo.png "Teaism") -->

<img src="logo.png" width="700">
<!-- <img src="logo.png" height="42" width="42"> -->

A minimalistic CUDA-based convolutional neural network library.

## Motivation

- Convolutional neural networks (CNNs) are at the core of computer vision applications recently
- Mobile/embedded platforms, e.g. quadrotors, demand fast and light-weighted CNN libraries. Modern deep learning libraries heavily depends on third-party libraries and hence are hard to be configured on mobile/embedded platforms (like Nvidia TX1). This effort aims at developing a full-fledged yet minimalistic CNN library that depends only on C++0x and CUDA 8.0.

| Library  | Dependencies |
| ------------- | ------------- |
| *Teaism*  | C/C++, CUDA  |
| Caffe | C/C++, CUDA, cuDNN, BLAS, Boost, Opencv, etc. |
| Tensorflow | C/C++, CUDA, cuDNN, Python, Bazel, Numpy, etc. |
| Torch | C/C++, CUDA, BLAS, LuaJIT, LuaRocks, OpenBLAS, etc. |

- For educational purposes :)

## Features

- 9 Layers implemented so as to reproduce LeNet, AlexNet, VGG, etc.
	- data, conv, fc, pooling, Relu, LRN, dropout, softmax, cross-entropy loss
- Model importer for importing trained Caffe models
- Forward inference / backpropagation
- Switching between CPU and GPU

## Directories
- basics/:  Major header files / base classes, e.g., session.hpp, layer.hpp, tensor.cu, etc.
- layers/:  All the layer implementations.
- tests/:  All test cases. It is recommended to browse demo_cifar10.cu, demo_mlp.cu, tests_alexnet.cu and tests_cifar10.cu to learn how to use this library.
- initializers/:  Parameter initialization for convolutional and fully connected layers.
- utils/:  Some utility functions.
- models/:  Scripts for training models in Caffe and importing trained models.

## Demos

- Import model and make inferences on Cifar10

```
$ make demo_cifar10 && ./demo_cifar10.o
Start demo cifar10 on GPU

datasets/cifar10/bmp_imgs/00006.bmp
network finished setup: 617.3 ms 
GPU memory usage: used = 346.250000, free = 7765.375000 MB, total = 8111.625000 MB
Loading weights ...
Loading conv: (5, 5, 3, 32): 
Loading bias: (1, 1, 1, 32): Loading conv: (5, 5, 32, 32): 
Loading bias: (1, 1, 1, 32): Loading conv: (5, 5, 32, 64): 
Loading bias: (1, 1, 1, 64): Loading fc: (1, 1, 64, 1024): 
Loading bias: (1, 1, 1, 64): Loading fc: (1, 1, 10, 64): 
Loading bias: (1, 1, 1, 10): data forward: 0.3 ms 
conv1 forward: 0.3 ms 
pool1 forward: 0.3 ms 
relu1 forward: 0.0 ms 
conv2 forward: 1.3 ms 
pool2 forward: 0.2 ms 
relu2 forward: 0.0 ms 
conv3 forward: 2.3 ms 
pool3 forward: 0.4 ms 
relu3 forward: 0.0 ms 
fc4 forward: 1.7 ms 
fc5 forward: 0.0 ms 
softmax forward: 0.1 ms 

Total forward time: 6.8 ms

Prediction: 
Airplane probability: 0.0000 
Automobile probability: 0.9993 
Bird probability: 0.0000 
Cat probability: 0.0000 
Deer probability: 0.0000 
Dog probability: 0.0000 
Frog probability: 0.0000 
Horse probability: 0.0005 
Ship probability: 0.0000 
Truck probability: 0.0001
```

- Multilayer perceptron

```
$ make demo_mlp && ./demo_mlp.cu
The example shows counting how many ones in the input: 
{0,0} -> {0,0,1} 
{0,1} -> {0,1,0} 
{1,0} -> {0,1,0} 
{1,1} -> {1,0,0}
Network: input(2) - fc(3) - fc(3) - softmax - cross_entropy_loss
input: 
0,1
0,0
1,0
1,1

ground truth: 
0 1 0
1 0 0
0 1 0
0 0 1

Training (learning rate = 0.1) .. 

-----iteration 5000-------
test input: 
0,0
1,0
1,1
0,1
out activations:
0.978394 0.021566 0.000040 
0.009701 0.878047 0.112252 
0.000000 0.101604 0.898396 
0.009701 0.878047 0.112252
```

working in progress .. 