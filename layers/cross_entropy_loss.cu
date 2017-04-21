
#ifndef CROSS_ENTROPY_LOSS_LAYER_CUH_
#define CROSS_ENTROPY_LOSS_LAYER_CUH_

#include <assert.h>
#include <math.h>
#include "basics/layer.hpp"
#include "basics/tensor.cu"
#include "basics/session.hpp"

// TODO: implement CUDA kernel for backward()

#define BLOCKDIM 32

namespace CrossEntropyGPUKernels {

  template <class Dtype>
  __global__ void ForwardGPU(Tensor<Dtype>* bottom_0, Tensor<Dtype>* bottom_1, Tensor<Dtype>* top) {
    assert(bottom_0->GetDims()[0] == bottom_1->GetDims()[0]);
    assert(bottom_0->GetDims()[1] == 1);
    assert(bottom_0->GetDims()[2] == 1);
    assert(bottom_1->GetDims()[1] == 1);
    assert(bottom_1->GetDims()[2] == 1);
    assert(bottom_1->GetDims()[3] == 1);
    assert(top->GetDims()[0] == 1);
    assert(top->GetDims()[1] == 1);
    assert(top->GetDims()[2] == 1);
    assert(top->GetDims()[3] == 1);

    size_t batch_size = bottom_0->GetDims()[0];
    Dtype loss = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      Dtype label = bottom_1->at(i,0,0,0);
      Dtype p = bottom_0->at(i,0,0,label);
      loss -= log(p);
    }
    top->at(0,0,0,0) = loss / batch_size;
  }

}

template <class Dtype>
class CrossEntropyLoss: public Layer<Dtype> {
public:
  CrossEntropyLoss(size_t filler): filler(filler) {}

  ~CrossEntropyLoss() {}

  void Forward(Tensor<Dtype>* bottom, Tensor<Dtype>* top) {}
  std::vector<Tensor<Dtype>* > Forward(const std::vector<Tensor<Dtype> *> &) {}
  void Forward(const std::vector<Tensor<Dtype>*> &, std::vector<Tensor<Dtype>*> &);
  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

  size_t filler;
private:

};

template <class Dtype>
void CrossEntropyLoss<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottom, std::vector<Tensor<Dtype>*> &top) {
  assert(bottom.size() == 2);  // Should have only two bottom tensors
  assert(top.size() == 1);  // Should have only one top tensor


  if (Session::GetSession()->gpu) {
    CrossEntropyGPUKernels::ForwardGPU<<<1, 1>>>(bottom[0], bottom[1], top[0]);
  } else {  
    assert(bottom[0]->GetDims()[0] == bottom[1]->GetDims()[0]);
    assert(bottom[0]->GetDims()[1] == 1);
    assert(bottom[0]->GetDims()[2] == 1);
    assert(bottom[1]->GetDims()[1] == 1);
    assert(bottom[1]->GetDims()[2] == 1);
    assert(bottom[1]->GetDims()[3] == 1);
    assert(top[0]->GetDims()[0] == 1);
    assert(top[0]->GetDims()[1] == 1);
    assert(top[0]->GetDims()[2] == 1);
    assert(top[0]->GetDims()[3] == 1);

    size_t batch_size = bottom[0]->GetDims()[0];

    Dtype loss = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      Dtype label = bottom[1]->at(i,0,0,0);
      Dtype p = bottom[0]->at(i,0,0,label);
      loss -= log(p);
    }
    top[0]->at(0,0,0,0) = loss / batch_size;
  }
}


#endif  // CROSS_ENTROPY_LOSS_LAYER_CUH_
