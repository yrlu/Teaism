
#ifndef CONV2D_LAYER_CUH_
#define CONV2D_LAYER_CUH_

#include <assert.h>
#include <stdio.h>
#include "basics/layer.hpp"
#include "basics/tensor.cu"
#include "basics/session.hpp"
#include "basics/initializer.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils/helper_cuda.h"
#include "utils/helper_string.h"
#include "initializers/gaussian_kernel_initializer.cu"
#include "utils/utils.cu"
#include "utils/computations.cu"
#include "basics/commons.hpp"

#define BLOCKDIM 32

// enum PADDING {SAME, VALID};

template <class Dtype>
class Conv2D: public Layer<Dtype> {
public:
  // use the same initializer to initialize W_ and b_
  Conv2D(size_t kernel_height, size_t kernel_width, size_t in_channels, 
    size_t out_channels, size_t stride, Initializer<Dtype>* initializer = NULL, PADDING _padding=SAME);

  ~Conv2D();

  void Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops);
  void Backward(const std::vector<Tensor<Dtype>*> &tops,
                        const std::vector<Tensor<Dtype>*> &tops_diff,
                        const std::vector<Tensor<Dtype>*> &bottoms,
                        const std::vector<Tensor<Dtype>*> &bottoms_diff);
  void GetTopsDims(const std::vector<size_t*> &bottoms_dims, 
                  const std::vector<size_t*> &tops_dims);

  const size_t kernel_height;
  const size_t kernel_width;
  const size_t in_channels;
  const size_t out_channels;
  const size_t stride;
  const PADDING padding;
  Tensor<Dtype>* W_;
  Tensor<Dtype>* b_;
  Tensor<Dtype>* W_flipped_;
  Tensor<Dtype>* W_diff_;
  Tensor<Dtype>* b_diff_;
private:
  const Initializer<Dtype>* initializer_;
  void InitParams();
  void InitDiffs();
  void FlipWeights();
};



template<class Dtype> 
Conv2D<Dtype>::Conv2D(size_t kernel_height, size_t kernel_width, size_t in_channels, 
    size_t out_channels, size_t stride, Initializer<Dtype>* initializer, PADDING _padding):
      kernel_height(kernel_height), kernel_width(kernel_width),
      in_channels(in_channels), out_channels(out_channels), 
      stride(stride), initializer_(initializer),
      padding(_padding), W_flipped_(NULL), W_diff_(NULL), b_diff_(NULL) {
  size_t w_dims[4] = {kernel_height, kernel_width, in_channels, out_channels};
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


template<class Dtype>
Conv2D<Dtype>::~Conv2D() {
  if (Session::GetSession()->gpu) {
    if (W_!= NULL) {
      cudaFree(W_);
      W_ = NULL;
    }
    if (W_flipped_ != NULL) {
      cudaFree(W_flipped_);
      W_flipped_ = NULL;
    }
    if (b_ != NULL) {
      cudaFree(b_);
      b_ = NULL;
    }
    if (W_diff_ != NULL) {
      cudaFree(W_diff_);
      W_diff_ = NULL;
    }
    if (b_diff_ != NULL) {
      cudaFree(b_diff_);
      b_diff_ = NULL;
    }
  } else {
    if(W_ != NULL) {
      delete W_;
      W_ = NULL;
    }
    if(W_flipped_ != NULL) { 
      delete W_flipped_;
      W_flipped_ = NULL;
    }
    if(b_ != NULL) {
      delete b_;
      b_ = NULL;
    }
    if(W_diff_ != NULL) {
      delete W_diff_;
      W_diff_ = NULL;
    }
    if(b_diff_ != NULL) {
      delete b_diff_; 
      b_diff_ = NULL;
    }
  }
}

template<class Dtype>
void Conv2D<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) {
  assert(bottoms.size()==1);
  assert(tops.size()==1);
  Tensor<Dtype> * bottom = bottoms[0];
  Tensor<Dtype> * top = tops[0];

  if (Session::GetSession()->gpu) {
    ComputationsGPU::ConvolutionGPU(bottom, top, W_, b_, stride, padding);
  } else {
    ComputationsCPU::ConvolutionCPU(bottom, top, W_, b_, stride, padding);
  }
}



template<class Dtype>
void Conv2D<Dtype>::GetTopsDims(const std::vector<size_t*> &bottoms_dims, 
                      const std::vector<size_t*> &tops_dims) {
  assert(bottoms_dims.size());
  assert(tops_dims.size());
  size_t * b_dims = bottoms_dims[0];
  size_t * t_dims = tops_dims[0];
  if(padding == SAME) {
    t_dims[0] = b_dims[0];
    t_dims[1] = b_dims[1]/stride;
    t_dims[2] = b_dims[2]/stride;
    t_dims[3] = out_channels;
  } else if(padding == VALID) {
    t_dims[0] = b_dims[0];
    t_dims[1] = b_dims[1]/stride - kernel_height + 1;
    t_dims[2] = b_dims[2]/stride - kernel_width + 1;
    t_dims[3] = out_channels;
  }
}

template<class Dtype> 
__global__ void FlipKernel(Tensor<Dtype> * W_, Tensor<Dtype> * W_flipped_) {
  size_t kernel_height = W_->GetDims()[0];
  size_t kernel_width = W_->GetDims()[1];
  size_t in_channels = W_->GetDims()[2];
  size_t out_channels = W_->GetDims()[3];
  for(int h = 0; h < kernel_height; h++) {
    for(int w = 0; w < kernel_width; w++) {
      for(int i = 0; i < in_channels; i++) {
        for(int o = 0; o < out_channels; o++) {
          W_flipped_->at(kernel_height-1-h, kernel_width-1-h, o, i) = W_->at(h, w, i, o);
          // printf("%f ", W_flipped_->at(kernel_height-1-h, kernel_width-1-h, o, i));
          printf("%f ", W_->at(h, w, i, o));
        }
        printf("\n");
      }
        printf("\n");
    }
        printf("\n");
  }
}

template<class Dtype>
void Conv2D<Dtype>::FlipWeights() {
  if(Session::GetSession()->gpu) {
    if(W_flipped_ == NULL) {
      size_t w_dims[4] = {kernel_height, kernel_width, out_channels, in_channels};
      W_flipped_ = Tensor<Dtype>::CreateTensorGPU(w_dims);
    }
    FlipKernel<Dtype><<<1,1>>>(W_, W_flipped_);
  } else {
    if(W_flipped_ == NULL) {
      size_t w_dims[4] = {kernel_height, kernel_width, out_channels, in_channels};
      W_flipped_ = Tensor<Dtype>::CreateTensorCPU(w_dims);
    }

    for(int h = 0; h < kernel_height; h++) {
      for(int w = 0; w < kernel_width; w++) {
        for(int i = 0; i < in_channels; i++) {
          for(int o = 0; o < out_channels; o++) {
            W_flipped_->at(kernel_height-1-h, kernel_width-1-h, o, i) = W_->at(h, w, i, o);
          }
        }
      }
    }
  }
}


template<class Dtype>
void Conv2D<Dtype>::InitParams() {
  if (initializer_!=NULL) {
    initializer_->Initialize(W_, b_, Session::GetSession()->gpu);
  } else {
    GaussianKernelInitializer<Dtype>((Dtype)(kernel_width+kernel_height)/2).Initialize(W_, b_, Session::GetSession()->gpu);
  }
}

template<class Dtype>
void Conv2D<Dtype>::InitDiffs() {
  if(Session::GetSession()->gpu) {
    if(W_diff_ == NULL) {
      size_t w_dims[4] = {kernel_height, kernel_width, in_channels, out_channels};
      W_diff_ = Tensor<Dtype>::CreateTensorGPU(w_dims);
    }
    if(b_diff_ == NULL) {
      size_t b_dims[4] = {1, 1, 1, out_channels};
      b_diff_ = Tensor<Dtype>::CreateTensorGPU(b_dims);
    }
  } else {
    if(W_diff_ == NULL) {
      size_t w_dims[4] = {kernel_height, kernel_width, in_channels, out_channels};
      W_diff_ = Tensor<Dtype>::CreateTensorCPU(w_dims);
    }
    if(b_diff_ == NULL) {
      size_t b_dims[4] = {1, 1, 1, out_channels};
      b_diff_ = Tensor<Dtype>::CreateTensorCPU(b_dims);
    }
  }
}

template<class Dtype>
__global__ void ComputeWDiffKernel(Tensor<Dtype> * bottom, Tensor<Dtype> * top_diff, Tensor<Dtype> * W_diff_, Tensor<Dtype> * b_diff_, const size_t stride, PADDING padding) {
  int b = (blockDim.x * blockIdx.x) + threadIdx.x;
  int i = (blockDim.y * blockIdx.y) + threadIdx.y;
  int o = (blockDim.z * blockIdx.z) + threadIdx.z;
  
  
  size_t bh = bottoms[0]->GetDims()[1];
  size_t bw = bottoms[0]->GetDims()[2];
  size_t th = tops_diff[0]->GetDims()[1];
  size_t tw = tops_diff[0]->GetDims()[2];
  
  for(int h = 0; h < kernel_height; h++) {
    for(int w = 0; w < kernel_width; w++) {
      Dtype sum = 0;
      if(padding == SAME) {
        for(int thi = 0; thi < th; thi++) {
          for(int twi = 0; twi < tw; twi++) {
            sum += bottoms[0]->atPadding(b, h-kernel_height/2 + thi, w - kernel_width/2 + twi, i) * tops_diff[0]->at(b, thi, twi, i);
          }
        }
      } else if(padding == VALID) {
        for(int thi = 0; thi < th; thi++) {
          for(int twi = 0; twi < tw; twi++) {
            sum += bottoms[0]->atPadding(b, h+thi, w+twi, i) * tops_diff[0]->at(b, thi, twi, i);
          }
        }
      }
      W_diff_->at(h, w, i, o) = sum;
    }
  }
}

template<class Dtype>
void Conv2D<Dtype>::Backward(const std::vector<Tensor<Dtype>*> &tops,
              const std::vector<Tensor<Dtype>*> &tops_diff,
              const std::vector<Tensor<Dtype>*> &bottoms,
              const std::vector<Tensor<Dtype>*> &bottoms_diff) {
  assert(bottoms.size() == 1);
  assert(bottoms_diff.size() == 1);
  assert(tops.size() == 1);
  assert(tops_diff.size() == 1);
  InitDiffs();
  FlipWeights();
  if(Session::GetSession()->gpu) {
    ComputationsGPU::ConvolutionGPU(tops_diff[0], bottoms_diff[0], W_flipped_, (Tensor<Dtype>*)NULL, stride, padding);
    // assumes stride = 1
        
    size_t batch_size = Session::GetSession()->batch_size;
    dim3 blocksInGrid(batch_size/BLOCKDIM+1, in_channels/8+1, out_channels/4+1);
    dim3 threadsPerBlock(BLOCKDIM, 8, 4);
    ComputeWDiffKernel<<<blocksInGrid, threadsPerBlock>>>(bottoms[0], tops_diff[0], W_diff_, b_diff_, stride, padding);
  } else {
    ComputationsCPU::ConvolutionCPU(tops_diff[0], bottoms_diff[0], W_flipped_, (Tensor<Dtype>*)NULL, stride, padding);
    // assumes stride = 1
    size_t batch_size = bottoms[0]->GetDims()[0];
    size_t bh = bottoms[0]->GetDims()[1];
    size_t bw = bottoms[0]->GetDims()[2];
    size_t th = tops[0]->GetDims()[1];
    size_t tw = tops[0]->GetDims()[2];
    
    for(int b = 0; b < batch_size; b++) {
      for(int i = 0; i < in_channels; i++) {
        for(int o = 0; o < out_channels; o++) {
          for(int h = 0; h < kernel_height; h++) {
            for(int w = 0; w < kernel_width; w++) {
              Dtype sum = 0;
              if(padding == SAME) {
                for(int thi = 0; thi < th; thi++) {
                  for(int twi = 0; twi < tw; twi++) {
                    sum += bottoms[0]->atPadding(b, h-kernel_height/2 + thi, w - kernel_width/2 + twi, i) * tops_diff[0]->at(b, thi, twi, i);
                  }
                }
              } else if(padding == VALID) {
                for(int thi = 0; thi < th; thi++) {
                  for(int twi = 0; twi < tw; twi++) {
                    sum += bottoms[0]->atPadding(b, h+thi, w+twi, i) * tops_diff[0]->at(b, thi, twi, i);
                  }
                }
              }
              W_diff_->at(h, w, i, o) = sum;
            }
          }
        }
      }
    }
  }
}

#endif  // CONV2D_LAYER_CUH_
