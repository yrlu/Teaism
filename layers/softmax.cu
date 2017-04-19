
#ifndef SOFTMAX_LAYER_CUH_
#define SOFTMAX_LAYER_CUH_

#include <assert.h>
#include <math.h>
#include "basics/layer.hpp"
#include "basics/tensor.cu"
#include "basics/session.hpp"

// TODO: implement CUDA kernel for backward()

#define BLOCKDIM 32

template <class Dtype>
__global__ void Forward_GPU(Tensor<Dtype>* bottom, Tensor<Dtype>* top) {
  size_t batch_idx = threadIdx.x;
  size_t batch_size = bottom->GetDims()[0];
  size_t nchannels = bottom->GetDims()[3];
  assert(batch_idx < batch_size);

  Dtype denominator = 0;
  for (size_t j = 0; j < nchannels; ++j) {
    top->at(batch_idx,0,0,j) = (Dtype) exp(bottom->at(batch_idx,0,0,j));
    denominator += top->at(batch_idx,0,0,j);
  }
  for (size_t j = 0; j < nchannels; ++j) {
    top->at(batch_idx,0,0,j) = top->at(batch_idx,0,0,j) / denominator;
  }
}

template <class Dtype>
class Softmax: public Layer<Dtype> {
public:
  Softmax(size_t filler):filler(filler) {}

  ~Softmax() {}

  void Forward(Tensor<Dtype>* bottom, Tensor<Dtype>* top) {}
  std::vector<Tensor<Dtype>* > Forward(const std::vector<Tensor<Dtype> *> &) {}
  void Forward(const std::vector<Tensor<Dtype>*> &, std::vector<Tensor<Dtype>*> &);

  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

  size_t filler;
private:

};

template <class Dtype>
void Softmax<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottom, std::vector<Tensor<Dtype>*> &top) {
  assert(bottom.size() == 1);  // Need only one bottom
  assert(bottom[0]->GetDims()[1] == 1);  // The dimension of the 2nd channel should be 1
  assert(bottom[0]->GetDims()[2] == 1);  // The dimension of the 3rd channel should be 1
  assert(bottom[0]->GetDims()[0] == top[0]->GetDims()[0]);  // bottom channel should be equal to top channel
  assert(bottom[0]->GetDims()[1] == top[0]->GetDims()[1]);
  assert(bottom[0]->GetDims()[2] == top[0]->GetDims()[2]);
  assert(bottom[0]->GetDims()[3] == top[0]->GetDims()[3]);

  size_t batch_size= bottom[0]->GetDims()[0];
  size_t nchannels = bottom[0]->GetDims()[3];

  if (Session::GetSession()->gpu) {
    Forward_GPU<<<1, batch_size>>>(bottom[0], top[0]); 
  } else {
    Dtype denominator;
    for (size_t i = 0; i < batch_size; ++i) {
      denominator = 0;
      for (size_t j = 0; j < nchannels; ++j) {
        top[0]->at(i,0,0,j) = (Dtype) exp(bottom[0]->at(i,0,0,j));
        denominator += top[0]->at(i,0,0,j);
      }
      for (size_t j = 0; j < nchannels; ++j) {
        top[0]->at(i,0,0,j) = top[0]->at(i,0,0,j) / denominator;
      }
    }
  }
}


#endif  // SOFTMAX_LAYER_CUH_
