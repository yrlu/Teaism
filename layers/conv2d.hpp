#ifndef CONV2D_LAYER_CUH_
#define CONV2D_LAYER_CUH_

#include <vector>
#include <assert.h>
#include "basics/layer.hpp"
#include "basics/tensor.hpp"
#include "basics/initializer.hpp"
#include "initializers/const_initializer.hpp"


// TODO: implement CUDA kernel for forward()
// TODO: implement CUDA kernel for backward()

template <class Dtype>
class Conv2D: public Layer<Dtype> {
public:
  // use the same initializer to initialize W_ and b_
  Conv2D(size_t kernel_height, size_t kernel_width, size_t in_channels, 
    size_t out_channels, size_t stride, Initializer<Dtype>* initializer = NULL):
      kernel_height(kernel_height), kernel_width(kernel_width),
      in_channels(in_channels), out_channels(out_channels), 
      stride(stride), initializer_(initializer), 
      W_(Tensor<Dtype>({kernel_height, kernel_width, in_channels})),
      b_(Tensor<Dtype>({out_channels})) {
    InitParams();
  }

  // directly pass in W & b
  Conv2D(size_t kernel_height, size_t kernel_width, size_t in_channels, 
    size_t out_channels, size_t stride, Tensor<Dtype>& W, Tensor<Dtype>& b):
      kernel_height(kernel_height), kernel_width(kernel_width),
      in_channels(in_channels), out_channels(out_channels), 
      stride(stride), W_(W), b_(b) {}

  ~Conv2D() {}

  void Forward(Tensor<Dtype> & bottom, Tensor<Dtype> & top) {
    // Assert dimensions (n, hei, wid, channel)
    assert(bottom.GetDims().size() == 4);
    assert(top.GetDims().size() == 4);
    // TODO: implement CPU convolution
    // TODO: implement GPU convolution
  }
  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

  const size_t kernel_height;
  const size_t kernel_width;
  const size_t in_channels;
  const size_t out_channels;
  const size_t stride;

private:
  Tensor<Dtype> W_;
  Tensor<Dtype> b_;
  const Initializer<Dtype>* initializer_;
  void InitParams() {
    if (initializer_!=NULL) {
      initializer_->Initialize(W_);
      initializer_->Initialize(b_);
    } else {
      ConstInitializer<Dtype>((Dtype)0.1).Initialize(W_);
      ConstInitializer<Dtype>((Dtype)0).Initialize(b_);
    }
  }
};

#endif  // CONV2D_LAYER_CUH_
