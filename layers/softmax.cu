
#ifndef SOFTMAX_LAYER_CUH_
#define SOFTMAX_LAYER_CUH_

#include <assert.h>
#include <math.h>
#include "basics/layer.hpp"
#include "basics/tensor.cu"
#include "basics/session.hpp"

// TODO: implement CUDA kernel for backward()

#define BLOCKDIM 32

namespace SoftmaxGPUKernels {

  template <class Dtype>
  __global__ void ForwardGPUKernel(Tensor<Dtype>* bottom, Tensor<Dtype>* top) {
    const int batch_idx = threadIdx.x;
    const int batch_size = int(bottom->GetDims()[0]);
    const int nchannels = int(bottom->GetDims()[3]);

    Dtype denominator = 0;
    for (int j = 0; j < nchannels; ++j) {
      top->at(batch_idx,0,0,j) = (Dtype) exp(bottom->at(batch_idx,0,0,j));
      denominator += top->at(batch_idx,0,0,j);
    }
    assert(denominator != 0);
    for (int j = 0; j < nchannels; ++j) {
      top->at(batch_idx,0,0,j) = top->at(batch_idx,0,0,j) / denominator;
    }
  }

  template <class Dtype>
  __global__ void ForwardGPU(Tensor<Dtype>* bottom, Tensor<Dtype>* top) {
    assert(bottom->GetDims()[1] == 1);  // The dimension of the 2nd channel should be 1
    assert(bottom->GetDims()[2] == 1);  // The dimension of the 3rd channel should be 1
    assert(bottom->GetDims()[0] == top->GetDims()[0]);  // bottom channel should be equal to top channel
    assert(bottom->GetDims()[1] == top->GetDims()[1]);
    assert(bottom->GetDims()[2] == top->GetDims()[2]);
    assert(bottom->GetDims()[3] == top->GetDims()[3]);

    SoftmaxGPUKernels::ForwardGPUKernel<Dtype> <<<1,bottom->GetDims()[0]>>>(bottom, top);
  }

}

template <class Dtype>
class Softmax: public Layer<Dtype> {
public:
  Softmax() {}

  ~Softmax() {}

  void Forward(const std::vector<Tensor<Dtype>*> &, const std::vector<Tensor<Dtype>*> &);

  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

private:
};

template <class Dtype>
void Softmax<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) {
  assert(bottoms.size() == 1);  // Need only one bottom tensor
  assert(tops.size() == 1);  // Need only one bottom tensor

  if (Session::GetSession()->gpu) {
    SoftmaxGPUKernels::ForwardGPU<Dtype><<<1, 1>>>(bottoms[0], tops[0]); 
  } else {
    assert(bottoms[0]->GetDims()[1] == 1);  // The dimension of the 2nd channel should be 1
    assert(bottoms[0]->GetDims()[2] == 1);  // The dimension of the 3rd channel should be 1
    assert(bottoms[0]->GetDims()[0] == tops[0]->GetDims()[0]);  // bottom channel should be equal to tops channel
    assert(bottoms[0]->GetDims()[1] == tops[0]->GetDims()[1]);
    assert(bottoms[0]->GetDims()[2] == tops[0]->GetDims()[2]);
    assert(bottoms[0]->GetDims()[3] == tops[0]->GetDims()[3]);

    const size_t batch_size = bottoms[0]->GetDims()[0];
    const size_t nchannels = bottoms[0]->GetDims()[3];

    Dtype denominator;
    for (int i = 0; i < batch_size; ++i) {
      denominator = 0;
      for (int j = 0; j < nchannels; ++j) {
        tops[0]->at(i,0,0,j) = (Dtype) exp(bottoms[0]->at(i,0,0,j));
        denominator += tops[0]->at(i,0,0,j);
      }
      for (int j = 0; j < nchannels; ++j) {
        tops[0]->at(i,0,0,j) = tops[0]->at(i,0,0,j) / denominator;
      }
    }
  }
}


#endif  // SOFTMAX_LAYER_CUH_
