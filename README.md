<!-- ![Teaism](logo.png "Teaism") -->

<img src="logo.png" width="700">
<!-- <img src="logo.png" height="42" width="42"> -->

A minimalistic convolutional neural network library for CUDA-based embedded and mobile devices.

## Dependencies

- C++0x
- CUDA >= 7.5
- CMake 3.5

## Files
```
.
├── CMakeLists.txt
├── Makefile
├── README.md
├── basics
│   ├── initializer.hpp
│   ├── layer.hpp
│   ├── session.hpp
│   └── tensor.cu
├── initializers
│   ├── const_initializer.cu
│   └── gaussian_kernel_initializer.cu
├── layers
│   ├── conv2d.cu
│   ├── conv2d1.cu
│   ├── conv2dopt.cu
│   ├── cross_entropy_loss.cu
│   ├── data.cu
│   ├── dropout.cu
│   ├── fc.cu
│   ├── lrn.cu
│   ├── pooling.cu
│   ├── relu.cu
│   └── softmax.cu
├── logo.png
├── main.cu
├── networks
│   └── lenet.hpp
├── perf
│   └── tf_lenet.py
├── tests
│   ├── tests_alexnet.cu
│   ├── tests_cifar10.cu
│   ├── tests_const_initializer.cu
│   ├── tests_conv2d.cu
│   ├── tests_conv2dopt.cu
│   ├── tests_cross_entropy_loss.cu
│   ├── tests_data.cu
│   ├── tests_dropout.cu
│   ├── tests_fc.cu
│   ├── tests_gaussian_initializer.cu
│   ├── tests_get_tops_dims.cu
│   ├── tests_lenet.cu
│   ├── tests_lenet_fc.cu
│   ├── tests_lrn.cu
│   ├── tests_pooling.cu
│   ├── tests_relu.cu
│   ├── tests_rng.cu
│   ├── tests_softmax.cu
│   └── tests_tensor.cu
└── utils
    ├── bitmap_image.hpp
    ├── helper_cuda.h
    ├── helper_string.h
    ├── load_model.hpp
    └── utils.cu
```
