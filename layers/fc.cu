#ifndef FC_LAYER_CUH_
#define FC_LAYER_CUH_

#include "initializers/const_initializer.cu"
#include "basics/layer.hpp"
#include "utils/utils.cu"
// #include <stdio.h>

#define BLOCKDIM 32

namespace FCGPUKernels {
  template <class Dtype> 
  __global__ void ForwardGPUShared(Tensor<Dtype> * bottom, Tensor<Dtype> * top, Tensor<Dtype> * W_, Tensor<Dtype> * b_) {
    size_t n = bottom->GetDims()[0];
    size_t in_channels = bottom->GetDims()[3];
    size_t out_channels = top->GetDims()[3];
    int bi = (blockDim.x * blockIdx.x) + threadIdx.x; // batch idx
    int o = (blockDim.y * blockIdx.y) + threadIdx.y;  // output idi

    extern __shared__ Dtype s[];

    Dtype* in = s;
    Dtype* w = &s[in_channels*BLOCKDIM];

    for(int j = threadIdx.y; j < in_channels; j+= BLOCKDIM) {
      if(bi < n) {
        in[threadIdx.x*in_channels + j] = bottom->at(bi, 0, 0, j);
      }
    }

    for(int j = threadIdx.x; j < in_channels; j+= BLOCKDIM) {
      if(o < out_channels) {
        w[threadIdx.y*in_channels + j] = W_->at(0,0, blockDim.y * blockIdx.y + threadIdx.y, j);
      }
    }
    __syncthreads();

    if (bi < 0 || bi >= n || o < 0 || o >= out_channels) {
      return;
    }

    Dtype sum = 0;
    for(int i = 0; i < in_channels; i++) {
      sum += in[threadIdx.x*in_channels + i] * w[threadIdx.y*in_channels+i];
    }
    sum += b_->at(0,0,0,o);
    top->at(bi,0,0,o) = sum;
  }

  template <class Dtype>
  __global__ void ForwardGPU(Tensor<Dtype> * bottom, Tensor<Dtype> * top, Tensor<Dtype> * W_, Tensor<Dtype> * b_) {
    size_t n = bottom->GetDims()[0];
    size_t in_channels = bottom->GetDims()[3];
    size_t out_channels = top->GetDims()[3];
    int bi = (blockDim.x * blockIdx.x) + threadIdx.x; // batch idx
    int o = (blockDim.y * blockIdx.y) + threadIdx.y;  // output idi
    
    if (bi < 0 || bi >= n || o < 0 || o >= out_channels) {
      return;
    }
    Dtype sum = 0;
    for(int i = 0; i < in_channels; i++) {
      sum += bottom->at(bi, 0, 0, i) * W_->at(0,0,o,i);
      // sum += bottom->at(bi, 0, 0, i) * w[GetIdx(w_dims, 0, o, i)];
    }
    sum += b_->at(0,0,0,o);
    top->at(bi,0,0,o) = sum;
  }
}


template <class Dtype>
class FC: public Layer<Dtype> {
public:

  FC(size_t in_channels, size_t out_channels, Initializer<Dtype>* initializer = NULL):
      in_channels(in_channels), out_channels(out_channels), initializer_(initializer) {
    size_t w_dims[4] = {1, 1, out_channels, in_channels};
    size_t b_dims[4] = {1, 1, 1, out_channels};
    if (Session::GetSession()->gpu) {
      W_ = Tensor<Dtype>::CreateTensorGPU(w_dims);
      b_ = Tensor<Dtype>::CreateTensorGPU(b_dims);
    } else {
      W_ = Tensor<Dtype>::CreateTensorCPU(w_dims);
      b_ = Tensor<Dtype>::CreateTensorCPU(b_dims);
    }
    InitParams();
  }

  ~FC() {
    if (Session::GetSession()->gpu) {
      if (W_!= NULL) {
        cudaFree(W_);
        W_ = NULL;
      }
      if (b_ != NULL) {
        cudaFree(b_);
        b_ = NULL;
      }
    } else {
      if(W_ != NULL) {
        delete W_;
        W_ = NULL;
      }
      if(b_ != NULL) {
        delete b_;
        b_ = NULL;
      }
    }
  }


  void Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) {
    assert(bottoms.size() == 1);
    assert(tops.size() == 1);
    Tensor<Dtype> * bottom = bottoms[0];
    Tensor<Dtype> * top = tops[0];

    if (Session::GetSession()->gpu) {
      size_t batch_size = Session::GetSession()->batch_size;
      dim3 blocksInGrid(batch_size / BLOCKDIM + 1, out_channels / BLOCKDIM + 1);
      dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);

      if (in_channels*2 < 384) {
        FCGPUKernels::ForwardGPUShared<<<blocksInGrid,threadsPerBlock, 2*in_channels*BLOCKDIM*sizeof(Dtype)>>>(bottom, top, W_, b_);
      } else {
        FCGPUKernels::ForwardGPU<<<blocksInGrid,threadsPerBlock>>>(bottom, top, W_, b_);
      }
    } else {
      for(int b = 0; b < bottom->GetDims()[0]; b++) {
        for(int o = 0; o < out_channels; o++) {
          Dtype sum = 0;
          for(int i = 0; i < in_channels; i++) {
            sum += bottom->at(b, 0, 0, i) * W_->at(0,0,o,i);
          }
          sum += b_->at(0,0,0,o);
          top->at(b,0,0,o) = sum;
        }
      }
    }
  }

  void GetTopsDims(const std::vector<size_t*> &bottoms_dims, 
                   const std::vector<size_t*> &tops_dims) {
    assert(bottoms_dims.size());
    assert(tops_dims.size());
    size_t * b_dims = bottoms_dims[0];
    assert(b_dims[1] == 1);
    assert(b_dims[2] == 1);
    size_t * t_dims = tops_dims[0];
    t_dims[0] = b_dims[0];
    t_dims[1] = 1;
    t_dims[2] = 1;
    t_dims[3] = out_channels;
  }


  const size_t in_channels;
  const size_t out_channels;

  Tensor<Dtype>* W_;
  Tensor<Dtype>* b_;
private:
  const Initializer<Dtype>* initializer_;
  void InitParams() {
    if (initializer_!=NULL) {
      initializer_->Initialize(W_, b_, Session::GetSession()->gpu);
    } else {
      ConstInitializer<Dtype>(1.0, 1.0).Initialize(W_, b_, Session::GetSession()->gpu);
    }
  }
};



#endif  // FC_LAYER_CUH_
