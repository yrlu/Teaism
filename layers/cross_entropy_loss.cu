
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
  CrossEntropyLoss() {}

  ~CrossEntropyLoss() {}

  void Forward(const std::vector<Tensor<Dtype>*> &, const std::vector<Tensor<Dtype>*> &);
  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

  void GetTopsDims(const std::vector<size_t*> &, const std::vector<size_t*> &);

private:

};

template <class Dtype>
void CrossEntropyLoss<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) {
  assert(bottoms.size() == 2);  // Should have only two bottom tensors
  assert(tops.size() == 1);  // Should have only one top tensor


  if (Session::GetSession()->gpu) {
    CrossEntropyGPUKernels::ForwardGPU<<<1, 1>>>(bottoms[0], bottoms[1], tops[0]);
  } else {  
    assert(bottoms[0]->GetDims()[0] == bottoms[1]->GetDims()[0]);
    assert(bottoms[0]->GetDims()[1] == 1);
    assert(bottoms[0]->GetDims()[2] == 1);
    assert(bottoms[1]->GetDims()[1] == 1);
    assert(bottoms[1]->GetDims()[2] == 1);
    assert(bottoms[1]->GetDims()[3] == 1);
    assert(tops[0]->GetDims()[0] == 1);
    assert(tops[0]->GetDims()[1] == 1);
    assert(tops[0]->GetDims()[2] == 1);
    assert(tops[0]->GetDims()[3] == 1);

    size_t batch_size = bottoms[0]->GetDims()[0];

    Dtype loss = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      Dtype label = bottoms[1]->at(i,0,0,0);
      Dtype p = bottoms[0]->at(i,0,0,label);
      loss -= log(p);
    }
    tops[0]->at(0,0,0,0) = loss / batch_size;
  }
}

template <class Dtype>
void CrossEntropyLoss<Dtype>::GetTopsDims(const std::vector<size_t*> &bottoms_dims, const std::vector<size_t*> &tops_dims) {
  assert(bottoms_dims.size() == 2);
  assert(tops_dims.size() == 1);

  tops_dims[0][0] = 1;
  tops_dims[0][1] = 1;
  tops_dims[0][2] = 1;
  tops_dims[0][3] = 1;
}

#endif  // CROSS_ENTROPY_LOSS_LAYER_CUH_
