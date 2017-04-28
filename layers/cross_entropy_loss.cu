
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

  template <class Dtype>
  __global__ void BackwardGPU (Tensor<Dtype>* top,
                               Tensor<Dtype>* top_diff,
                               Tensor<Dtype>* bottom_0,
                               Tensor<Dtype>* bottom_1,
                               Tensor<Dtype>* bottom_diff_0) {
    int batch_idx = threadIdx.x;
    for (int j = 0; j < bottom_0->GetDims()[3]; ++j) {
      bottom_diff_0->at(batch_idx,0,0,j) = 0;
    }
    Dtype label = bottom_1->at(batch_idx,0,0,0);
    Dtype p = bottom_0->at(batch_idx,0,0,label);
    bottom_diff_0->at(batch_idx,0,0,label) = top_diff->at(0,0,0,0)/(p+0.0001);
  }

}

template <class Dtype>
class CrossEntropyLoss: public Layer<Dtype> {
public:
  CrossEntropyLoss() {}

  ~CrossEntropyLoss() {}

  void Forward(const std::vector<Tensor<Dtype>*>&, const std::vector<Tensor<Dtype>*>&);
  void Backward(const std::vector<Tensor<Dtype>*>& , const std::vector<Tensor<Dtype>*>&,
                const std::vector<Tensor<Dtype>*>&, const std::vector<Tensor<Dtype>*>&);

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
void CrossEntropyLoss<Dtype>::Backward (const std::vector<Tensor<Dtype>*>& tops, 
                                       const std::vector<Tensor<Dtype>*>& tops_diff,
                                       const std::vector<Tensor<Dtype>*>& bottoms,
                                       const std::vector<Tensor<Dtype>*>& bottoms_diff) {
  assert(tops.size() == 1);
  assert(tops_diff.size() == 1);
  assert(bottoms.size() == 2);
  assert(bottoms_diff.size() == 2);

  Tensor<Dtype>* top = tops[0];
  Tensor<Dtype>* top_diff = tops_diff[0];
  Tensor<Dtype>* bottom_0 = bottoms[0];
  Tensor<Dtype>* bottom_1 = bottoms[1];
  Tensor<Dtype>* bottom_diff_0 = bottoms_diff[0];
  // Not backpropagate to labels

  Session* S = Session::GetSession();
  int batch_size = S->batch_size;
  if (S->gpu) {
    CrossEntropyGPUKernels::BackwardGPU<Dtype><<<1,batch_size>>>(top,top_diff,bottom_0,bottom_1,bottom_diff_0);
  } else {
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < bottom_0->GetDims()[3]; ++j) {
        bottom_diff_0->at(i,0,0,j) = 0;
      }
      Dtype label = bottom_1->at(i,0,0,0);
      Dtype p = bottom_0->at(i,0,0,label);
      bottom_diff_0->at(i,0,0,label) = top_diff->at(0,0,0,0)/(p+0.0001);
    }
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
