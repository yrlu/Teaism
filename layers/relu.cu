#ifndef RELU_LAYER_CUH_
#define RELU_LAYER_CUH_

#include <assert.h>
#include "basics/layer.hpp"
#include "basics/tensor.cu"
#include "basics/session.hpp"

#define BLOCKDIM 32

namespace ReluGPUKernels {
  template<class Dtype>
  __global__ void ForwardGPU3(Tensor<Dtype> * bottom, Tensor<Dtype> * top) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    Dtype* b_data = bottom->GetDataPtr();
    Dtype* t_data = top->GetDataPtr();
    if(idx < 0 || idx >= bottom->size()) return;
    t_data[idx] = (b_data[idx] >= 0 ? b_data[idx] : 0);
  }

  template <class Dtype>
  __global__ void ForwardGPU(Tensor<Dtype> * bottom, Tensor<Dtype> * top) {
    size_t size = bottom->size();    
    ReluGPUKernels::ForwardGPU3<Dtype><<<size/(BLOCKDIM*BLOCKDIM) + 1, BLOCKDIM*BLOCKDIM>>>(bottom, top);
  }
  template <class Dtype>
  __global__ void BackwardGPU(Tensor<Dtype>* top, Tensor<Dtype>* top_diff,
                              Tensor<Dtype>* bottom, Tensor<Dtype>* bottom_diff) {
    int batch_idx = threadIdx.x;
    for (int h = 0; h < top->GetDims()[1]; ++h) {
      for (int w = 0; w < top->GetDims()[2]; ++w) {
        for (int c = 0; c < top->GetDims()[3]; ++c) {
          if (bottom->at(batch_idx,h,w,c) <= 0) {
            bottom_diff->at(batch_idx,h,w,c) = 0;
          } else {
            bottom_diff->at(batch_idx,h,w,c) = top_diff->at(batch_idx,h,w,c);
          }
        }
      }
    }
  }
}

template <class Dtype>
class Relu: public Layer<Dtype> {
public:
  void GetTopsDims(const std::vector<size_t*> &bottoms_dims, 
                   const std::vector<size_t*> &tops_dims);

  void Forward(const std::vector<Tensor<Dtype>*> &bottoms, 
               const std::vector<Tensor<Dtype>*> &tops);

  void Backward(const std::vector<Tensor<Dtype>*> &tops, 
                const std::vector<Tensor<Dtype>*> &tops_diff,
                const std::vector<Tensor<Dtype>*> &bottoms,
                const std::vector<Tensor<Dtype>*> &bottoms_diff);

};

template<class Dtype>
void Relu<Dtype>::GetTopsDims(const std::vector<size_t*> &bottoms_dims, 
                   const std::vector<size_t*> &tops_dims) {
  assert(bottoms_dims.size());
  assert(tops_dims.size());
  size_t * b_dims = bottoms_dims[0];
  size_t * t_dims = tops_dims[0];
  t_dims[0] = b_dims[0];
  t_dims[1] = b_dims[1];
  t_dims[2] = b_dims[2];
  t_dims[3] = b_dims[3];
}

template<class Dtype>
void Relu<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottoms, 
                          const std::vector<Tensor<Dtype>*> &tops) {
  assert(bottoms.size()==1);
  assert(tops.size()==1);
  Tensor<Dtype> * bottom = bottoms[0];
  Tensor<Dtype> * top = tops[0];

  if (Session::GetSession()->gpu) {
    ReluGPUKernels::ForwardGPU<<<1,1>>>(bottom, top);
  } else {
    for(int b = 0; b < bottom->GetDims()[0]; b++) {
      for(int o = 0; o < bottom->GetDims()[3]; o++) {
        for(int x = 0; x < bottom->GetDims()[2]; x += 1) {
          for(int y = 0; y < bottom->GetDims()[1]; y += 1) {
            Dtype val = bottom->at(b, y, x, o);
            top->at(b, y, x, o) = (val >= 0 ? val : 0);
          }
        }
      }
    }
  }
}

template <class Dtype>
void Relu<Dtype>::Backward(const std::vector<Tensor<Dtype>*> &tops,
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
    ReluGPUKernels::BackwardGPU<Dtype><<<1,batch_size>>>(top, top_diff, bottom, bottom_diff);
  } else {
    for (int b = 0; b < top->GetDims()[0]; ++b) {
      for (int h = 0; h < top->GetDims()[1]; ++h) {
        for (int w = 0; w < top->GetDims()[2]; ++w) {
          for (int c = 0; c < top->GetDims()[3]; ++c) {
            if (bottom->at(b,h,w,c) <= 0) { 
              bottom_diff->at(b,h,w,c) = 0;
            } else {
              bottom_diff->at(b,h,w,c) = top_diff->at(b,h,w,c);
            }
          }
        }
      }
    }
  }

}


#endif 
