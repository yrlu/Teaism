
#ifndef CONV2D_LAYER_CUH_
#define CONV2D_LAYER_CUH_

#include <assert.h>
#include <stdio.h>
#include <string>
#include "basics/layer.hpp"
#include "basics/tensor.cu"
#include "basics/session.hpp"
#include "cuda_runtime.h"
#include "utils/helper_cuda.h"
#include "utils/helper_string.h"

// TODO: implement CUDA kernel for backward()

#define BLOCKDIM 32


template <class Dtype>
class Data: public Layer<Dtype> {
public:
  // use the same initializer to initialize W_ and b_
  Data(unsigned batch_size, std::string img_list):
      batch_size(batch_size), img_list(img_list) {
    if (Session::GetSession()->gpu) {
      W_ = Tensor<Dtype>::CreateTensorGPU(w_dims);
      b_ = Tensor<Dtype>::CreateTensorGPU(b_dims);
    } else {
      W_ = Tensor<Dtype>::CreateTensorCPU(w_dims);
      b_ = Tensor<Dtype>::CreateTensorCPU(b_dims);
    }
    InitParams();
  }

  ~Conv2D() {
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

  void Forward(Tensor<Dtype> * bottom, Tensor<Dtype> * top) {
    if (Session::GetSession()->gpu) {
      ForwardGPU<<<1,1>>>(top, batch_size, img_list);
    } else {
      ForwardCPU(top, batch_size, img_list);
  }

  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

  const size_t batch_size;
  const std::string img_list;

private:
  Tensor<Dtype>* W_;
  Tensor<Dtype>* b_;
  const Initializer<Dtype>* initializer_;
  void InitParams() {
    if (initializer_!=NULL) {
      initializer_->Initialize(W_, b_);
    } else {
      GaussianKernelInitializer<Dtype>((Dtype)5.0).Initialize(W_, b_, Session::GetSession()->gpu);
    }
  }
};


#endif  // CONV2D_LAYER_CUH_
