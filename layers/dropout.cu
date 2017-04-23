#ifndef DROPOUT_LAYER_CUH_
#define DROPOUT_LAYER_CUH_

#include "basics/layer.hpp"
#include "assert.h"


namespace DropoutGPUKernels {

  template <class Dtype>
  __global__ void ForwardGPUKernel(Tensor<Dtype>* bottom, Tensor<Dtype>* top, float scale_, int batch_idx) {
    int channel_idx = threadIdx.x;
    for (int i = 0; i < bottom->GetDims()[1]; ++i) {
      for (int j = 0; j < bottom->GetDims()[2]; ++j) {
        top->at(batch_idx, i, j, channel_idx) = scale_ * bottom->at(batch_idx, i, j, channel_idx);
      }
    }
  }

  template <class Dtype>
  __global__ void ForwardGPU(Tensor<Dtype>* bottom, Tensor<Dtype>* top, float scale_) {
    assert(bottom->GetDims()[0] == top->GetDims()[0]);
    assert(bottom->GetDims()[1] == top->GetDims()[1]);
    assert(bottom->GetDims()[2] == top->GetDims()[2]);
    assert(bottom->GetDims()[3] == top->GetDims()[3]);

    int batch_idx = threadIdx.x;
    DropoutGPUKernels::ForwardGPUKernel<Dtype><<<1, bottom->GetDims()[3]>>>(bottom, top, scale_, batch_idx);
  }

}

template <class Dtype>
class Dropout: public Layer<Dtype> {
public:

  Dropout(float p = 0.5, size_t seed = NULL);

  ~Dropout() {}

  void Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops);   
  
  void GetTopsDims(const std::vector<size_t*> &bottoms_dims, const std::vector<size_t*> &tops_dims);


  float keep_prob;

private:
  float scale_;
};

template <class Dtype>
Dropout<Dtype>::Dropout(float p, size_t seed): keep_prob(p) {
  if (Session::GetSession()->test) {
    scale_ = 1. / (1. - keep_prob);
  } else {
    // TODO: Set up random seed for training.
  }
}

template <class Dtype>
void Dropout<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) {
  assert(bottoms.size() == 1);  // Need one bottom tensor
  assert(tops.size() == 1);  // Need one top tensor

  Session* session = Session::GetSession();
  if (session->test) {
    if (session->gpu) {
      DropoutGPUKernels::ForwardGPU<Dtype><<<1,session->batch_size>>>(bottoms[0], tops[0], scale_);
    } else {
      assert(bottoms[0]->GetDims()[0] == tops[0]->GetDims()[0]);
      assert(bottoms[0]->GetDims()[1] == tops[0]->GetDims()[1]);
      assert(bottoms[0]->GetDims()[2] == tops[0]->GetDims()[2]);
      assert(bottoms[0]->GetDims()[3] == tops[0]->GetDims()[3]);

      for (size_t i = 0; i < bottoms[0]->GetDims()[0]; i++) {
        for (size_t j = 0; j < bottoms[0]->GetDims()[1]; j++) {
          for (size_t k = 0; k < bottoms[0]->GetDims()[2]; k++) {
            for (size_t l = 0; l < bottoms[0]->GetDims()[3]; l++) {
              tops[0]->at(i,j,k,l) = scale_ * bottoms[0]->at(i,j,k,l);
            }
          }
        }
      }
    }
  } else {
    // TODO: Implement dropout in training phase.
  }
}

template <class Dtype>
void Dropout<Dtype>::GetTopsDims(const std::vector<size_t*> &bottoms_dims, const std::vector<size_t*> &tops_dims) {
  assert(bottoms_dims.size() == 1);
  assert(tops_dims.size() == 1);

  tops_dims[0][0] = bottoms_dims[0][0];
  tops_dims[0][1] = bottoms_dims[0][1];
  tops_dims[0][2] = bottoms_dims[0][2];
  tops_dims[0][3] = bottoms_dims[0][3];
}



#endif  // DROPOUT_LAYER_CUH_
