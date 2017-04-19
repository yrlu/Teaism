
#ifndef SOFTMAX_LOSS_LAYER_CUH_
#define SOFTMAX_LOSS_LAYER_CUH_

#include <assert.h>
#include "basics/layer.hpp"
#include "basics/tensor.cu"
#include "basics/session.hpp"

// TODO: implement CUDA kernel for backward()

#define BLOCKDIM 32


template <class Dtype>
class SoftmaxLoss: public Layer<Dtype> {
public:
  SoftmaxLoss() {}

  ~SoftmaxLoss() {}

  void Forward(Tensor<Dtype>* bottom, Tensor<Dtype>* top) {}
  std::vector<Tensor<Dtype>* > Forward(const std::vector<Tensor<Dtype> *> &);

  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

private:

};



#endif  // SOFTMAX_LOSS_LAYER_CUH_
