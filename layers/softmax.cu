
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
__global__ void Forward_GPU(Tensor<Dtype>* bottom) {


}

template <class Dtype>
class Softmax: public Layer<Dtype> {
public:
  Softmax() {}

  ~Softmax() {}

//  void Forward(Tensor<Dtype>* bottom, Tensor<Dtype>* top) {}
//  std::vector<Tensor<Dtype>* > Forward(const std::vector<Tensor<Dtype> *> &) {}
  void Forward(const std::vector<Tensor<Dtype>*> &, std::vector<Tensor<Dtype>*> &);

  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

private:

};

template <class Dtype>
void Softmax<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottom, std::vector<Tensor<Dtype>*> &top) {
  assert(bottom.size() == 1);  // Need only one bottom
  assert(bottom[0]->GetDims()[1] == 1);  // The dimension of the 2nd channel should be 1
  assert(bottom[0]->GetDims()[2] == 1);  // The dimension of the 3rd channel should be 1
  assert(bottom.GetDims() == top.GetDims());

  if (Session::GetSession()->gpu) {
    Forward_GPU<<<1, bottom[0]->GetDims()[0]>>>(bottom[0]); 
  } else {
    Dtype denominator;
    for (size_t i = 0; i < bottom[0]->GetDims()[0]; ++i) {
      denominator = 0;
      for (size_t j = 0; j < bottom[0]->GetDims()[3]; ++j) {
        top[0]->at(i,0,0,j) = (Dtype) exp(bottom[0]->at(i,0,0,j));
        denominator += top[0]->at(i,0,0,j);
      }
      for (size_t j = 0; j < bottom[0]->GetDims()[3]; ++j) {
        top[0]->at(i,0,0,j) /= denominator;
      }
    }
  }
}


#endif  // SOFTMAX_LAYER_CUH_
