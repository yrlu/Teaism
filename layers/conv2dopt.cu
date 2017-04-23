
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

// TODO: implement CUDA kernel for backward()

#define BLOCKDIM 32

enum PADDING {SAME, VALID};

namespace ConvGPUKernels {

  template <class Dtype>
  __global__ void ForwardGPUKernel(Tensor<Dtype> * bottom, Tensor<Dtype> * top, Tensor<Dtype> * W, Tensor<Dtype> * b, int bi, int o, int stride, PADDING padding) {
    // bi is the index of the tensor
    // o is the output channel
    int x_top = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y_top = (blockDim.y * blockIdx.y) + threadIdx.y;
    size_t kernel_height = W->GetDims()[0];
    size_t kernel_width = W->GetDims()[1];
    int x = x_top*stride;
    int y = y_top*stride;

    if (!bottom->isValidIdx(bi, y, x, o) || !top->isValidIdx(bi, y_top, x_top, o)) {
      return;
    }

    if (padding==VALID) {
      x = kernel_width/2 + x_top*stride;
      y = kernel_height/2 + y_top*stride;
      if (!bottom->isValidIdx(bi, y, x, o) || !top->isValidIdx(bi, y_top, x_top, o) || !bottom->isValidIdx(bi, y + kernel_height/2, x + kernel_height/2, o)) {
        return;
      }
    }

    int idx[4] = {bi, y, x, o};
    size_t in_channels = bottom->GetDims()[3];
    Dtype sum = 0.0;
    for(int c = 0; c < in_channels; c++) {
      for(int i = 0; i < kernel_height; i++) {
        for(int j = 0; j < kernel_width; j++) {
          // (n, hei, wid, channel),   // (hei, wid, input, output)
          sum += bottom->atPadding(idx[0], idx[1]+i-int(kernel_height/2), idx[2]+j-int(kernel_width/2), c) * W->at(i, j, c, idx[3]);
        }
      }
    }
    sum += b->at(0, 0, 0, 0);
    top->at(bi, y_top, x_top, o) = sum;
  }

  template <class Dtype>
  __global__ void ForwardGPU(Tensor<Dtype> * bottom, Tensor<Dtype> * top, Tensor<Dtype> * W_, Tensor<Dtype> * b_, int stride, PADDING padding=SAME) {
    size_t n = bottom->GetDims()[0];
    size_t out_channels = top->GetDims()[3];
    size_t hei = top->GetDims()[1];
    size_t wid = top->GetDims()[2];
  
    dim3 blocksInGrid(wid / BLOCKDIM + 1, hei / BLOCKDIM + 1);
    dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
    for (int b = 0; b < n; b++) {
      for (int o = 0; o < out_channels; o++) {
        ConvGPUKernels::ForwardGPUKernel<Dtype><<<blocksInGrid, threadsPerBlock>>>(bottom, top, W_, b_, b, o, stride, padding);
      }
    }
  }
}

template <class Dtype>
class Conv2D: public Layer<Dtype> {
public:
  // use the same initializer to initialize W_ and b_
  Conv2D(size_t kernel_height, size_t kernel_width, size_t in_channels, 
    size_t out_channels, size_t stride, Initializer<Dtype>* initializer = NULL, PADDING _padding=SAME);

  ~Conv2D();

  void Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops);

  void GetTopsDims(const std::vector<size_t*> &bottoms_dims, 
                  const std::vector<size_t*> &tops_dims);

  const size_t kernel_height;
  const size_t kernel_width;
  const size_t in_channels;
  const size_t out_channels;
  const size_t stride;
  const PADDING padding;
private:
  Tensor<Dtype>* W_;
  Tensor<Dtype>* b_;
  const Initializer<Dtype>* initializer_;
  void InitParams(); 
};



template<class Dtype> 
Conv2D<Dtype>::Conv2D(size_t kernel_height, size_t kernel_width, size_t in_channels, 
    size_t out_channels, size_t stride, Initializer<Dtype>* initializer, PADDING _padding):
      kernel_height(kernel_height), kernel_width(kernel_width),
      in_channels(in_channels), out_channels(out_channels), 
      stride(stride), initializer_(initializer),
      padding(_padding) {
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

template<class Dtype>
void Conv2D<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) {
  assert(bottoms.size()==1);
  assert(tops.size()==1);
  Tensor<Dtype> * bottom = bottoms[0];
  Tensor<Dtype> * top = tops[0];

  if (Session::GetSession()->gpu) {
    ConvGPUKernels::ForwardGPU<<<1,1>>>(bottom, top, W_, b_, stride, padding);
  } else {
    for(int b = 0; b < bottom->GetDims()[0]; b++) {
      for(int o = 0; o < out_channels; o++) {
        if(padding==SAME) {
          for(int x = 0, x_top = 0; x < bottom->GetDims()[2]; x += stride, x_top += 1) {
            for(int y = 0, y_top = 0; y < bottom->GetDims()[1]; y += stride, y_top += 1) {
              // batch idx b, output layer o, pixel (x, y)
              // top->at({b, y, x, o}) = 
              int idx[4] = {b, y, x, o};
              Dtype sum = 0.0;
              for(int c = 0; c < in_channels; c++) {
                for(int i = 0; i < kernel_height; i++) {
                  for(int j = 0; j < kernel_width; j++) {
                    // (n, hei, wid, channel),   // (hei, wid, input, output)
                    int b_idx[4] = {idx[0], idx[1]+i-int(kernel_height/2), idx[2]+j-int(kernel_width/2), c};
                    int t_idx[4] = {i, j, c, idx[3]};
                    sum += bottom->atPadding(b_idx) * W_->at(t_idx);
                  }
                }
              }
              int b_idx[4] = {0,0,0,0};
              sum += b_->at(b_idx);
              int t_idx[4] = {b, y_top, x_top, o};
              
              top->at(t_idx) = sum;
            }
          }
        } else if (padding==VALID) {
          for(int x = kernel_width/2, x_top = 0; x < bottom->GetDims()[2] - kernel_width/2; x += stride, x_top += 1) {
            for(int y = kernel_height/2, y_top = 0; y < bottom->GetDims()[1] - kernel_height/2; y += stride, y_top += 1) {
              // batch idx b, output layer o, pixel (x, y)
              // top->at({b, y, x, o}) = 
              int idx[4] = {b, y, x, o};
              Dtype sum = 0.0;
              for(int c = 0; c < in_channels; c++) {
                for(int i = 0; i < kernel_height; i++) {
                  for(int j = 0; j < kernel_width; j++) {
                    // (n, hei, wid, channel),   // (hei, wid, input, output)
                    int b_idx[4] = {idx[0], idx[1]+i-int(kernel_height/2), idx[2]+j-int(kernel_width/2), c};
                    int t_idx[4] = {i, j, c, idx[3]};
                    sum += bottom->atPadding(b_idx) * W_->at(t_idx);
                  }
                }
              }
              int b_idx[4] = {0,0,0,0};
              sum += b_->at(b_idx);
              int t_idx[4] = {b, y_top, x_top, o};        
              top->at(t_idx) = sum;
            }
          }
        }
      }
    }
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
void Conv2D<Dtype>::InitParams() {
  if (initializer_!=NULL) {
    initializer_->Initialize(W_, b_, Session::GetSession()->gpu);
  } else {
    GaussianKernelInitializer<Dtype>((Dtype)(kernel_width+kernel_height)/2).Initialize(W_, b_, Session::GetSession()->gpu);
  }
}


#endif  // CONV2D_LAYER_CUH_
