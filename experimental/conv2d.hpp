#ifndef CONV2D_LAYER_CUH_
#define CONV2D_LAYER_CUH_

#include "layer.hpp"
#include "tensor.hpp"

// TODO: implement CUDA kernel for forward()
// TODO: implement CUDA kernel for backward()

template <class Dtype>
class Conv2D: public Layer<Dtype> {
public:
  Conv2D(size_t kernel_height, size_t kernel_width, size_t in_channels, 
    size_t out_channels, size_t stride):
      kernel_height(kernel_height), kernel_width(kernel_width),
      in_channels(in_channels), out_channels(out_channels), 
      stride(stride) {}
  ~Conv2D() {}

  void Forward(Tensor<Dtype> & bottom, Tensor<Dtype> & top) {}
  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

  const size_t kernel_height;
  const size_t kernel_width;
  const size_t in_channels;
  const size_t out_channels;
  const size_t stride;
};

#endif  // CONV2D_LAYER_CUH_