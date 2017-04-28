
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

    Dtype max_value = 0;
    for (int j = 0; j < nchannels; ++j) {
      if (bottom->at(batch_idx,0,0,j) > max_value) {
        max_value = bottom->at(batch_idx,0,0,j);
      }
    }

    Dtype denominator = 0;
    for (int j = 0; j < nchannels; ++j) {
      top->at(batch_idx,0,0,j) = (Dtype) exp(bottom->at(batch_idx,0,0,j)-max_value);
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

  template <class Dtype>
  __global__ void BackwardGPU(Tensor<Dtype>* top, Tensor<Dtype>* top_diff,
                              Tensor<Dtype>* bottom, Tensor<Dtype>* bottom_diff) {
    int batch_idx = threadIdx.x;
    int nchannels = top->GetDims()[3];
    for (int i = 0; i < nchannels; ++i) {
      bottom_diff->at(batch_idx,0,0,i) = 0;
      for (int j = 0; j < nchannels; ++j) {
        if (i==j) {
          bottom_diff->at(batch_idx,0,0,i) += 
            top->at(batch_idx,0,0,i) * (1-top->at(batch_idx,0,0,j)) * top_diff->at(batch_idx,0,0,j);
        } else {
          bottom_diff->at(batch_idx,0,0,i) -= 
            top->at(batch_idx,0,0,i) * top->at(batch_idx,0,0,j) * top_diff->at(batch_idx,0,0,j);
        }
      }
    }
  }

}

template <class Dtype>
class Softmax: public Layer<Dtype> {
public:
  Softmax() {}

  ~Softmax() {}

  void Forward(const std::vector<Tensor<Dtype>*>&, const std::vector<Tensor<Dtype>*>&);

  void Backward(const std::vector<Tensor<Dtype>*>&, const std::vector<Tensor<Dtype>*>&,
                const std::vector<Tensor<Dtype>*>&, const std::vector<Tensor<Dtype>*>&);

  void GetTopsDims(const std::vector<size_t*> &, const std::vector<size_t*> &); 

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
    Dtype max_value;
    for (int i = 0; i < batch_size; ++i) {
      max_value = 0;
      for (int j = 0; j < nchannels; ++j) {
        if (bottoms[0]->at(i,0,0,j) > max_value) {
          max_value = bottoms[0]->at(i,0,0,j);
        }
      }

      denominator = 0;
      for (int j = 0; j < nchannels; ++j) {
        tops[0]->at(i,0,0,j) = (Dtype) exp(bottoms[0]->at(i,0,0,j)-max_value);
        denominator += tops[0]->at(i,0,0,j);
      }
      for (int j = 0; j < nchannels; ++j) {
        tops[0]->at(i,0,0,j) = tops[0]->at(i,0,0,j) / denominator;
      }
    }
  }
}

template <class Dtype>
void Softmax<Dtype>::Backward(const std::vector<Tensor<Dtype>*> &tops, 
                              const std::vector<Tensor<Dtype>*> &tops_diff,
                              const std::vector<Tensor<Dtype>*> &bottoms,
                              const std::vector<Tensor<Dtype>*> &bottoms_diff) {
  assert(tops.size() == 1);
  assert(tops_diff.size() == 1);
  assert(bottoms.size() == 1);
  assert(bottoms_diff.size() == 1);

  Tensor<Dtype>* top = tops[0];
  Tensor<Dtype>* top_diff = tops_diff[0];
  Tensor<Dtype>* bottom = bottoms[0];
  Tensor<Dtype>* bottom_diff = bottoms_diff[0];

  Session* S = Session::GetSession();
  int batch_size = S->batch_size;
  if (S->gpu) {
    SoftmaxGPUKernels::BackwardGPU<Dtype><<<1,batch_size>>>(top,top_diff,bottom,bottom_diff);
  } else {
    for (int b = 0; b < batch_size; ++b) {
      int nchannels = top->GetDims()[3];
      for (int i = 0; i < nchannels; ++i) {
        bottom_diff->at(b,0,0,i) = 0;
        for (int j = 0; j < nchannels; ++j) {
          if (i==j) {
            bottom_diff->at(b,0,0,i) += top->at(b,0,0,i) * (1-top->at(b,0,0,j)) * top_diff->at(b,0,0,j);
          } else {
            bottom_diff->at(b,0,0,i) -= top->at(b,0,0,i) * top->at(b,0,0,j) * top_diff->at(b,0,0,j);
          }
        }
      }
    }
  }

}


template <class Dtype>
void Softmax<Dtype>::GetTopsDims(const std::vector<size_t*> &bottoms_dims, const std::vector<size_t*> &tops_dims) {
  assert(bottoms_dims.size() == 1);
  assert(tops_dims.size() == 1);

  tops_dims[0][0] = bottoms_dims[0][0];
  tops_dims[0][1] = bottoms_dims[0][1];
  tops_dims[0][2] = bottoms_dims[0][2];
  tops_dims[0][3] = bottoms_dims[0][3];
}


#endif  // SOFTMAX_LAYER_CUH_
