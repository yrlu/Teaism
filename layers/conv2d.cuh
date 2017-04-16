#ifndef CONV2D_LAYER_CUH_
#define CONV2D_LAYER_CUH_

#include <vector>
#include <assert.h>
#include "basics/layer.hpp"
#include "basics/tensor.hpp"
#include "basics/session.hpp"
#include "basics/initializer.hpp"
#include "initializers/gaussian_kernel_initializer.cuh"


// TODO: implement CUDA kernel for forward()
// TODO: implement CUDA kernel for backward()

template <class Dtype>
class Conv2D: public Layer<Dtype> {
public:
  // use the same initializer to initialize W_ and b_
  Conv2D(size_t kernel_height, size_t kernel_width, size_t in_channels, 
    size_t out_channels, size_t stride, Initializer<Dtype>* initializer = NULL):
      kernel_height(kernel_height), kernel_width(kernel_width),
      in_channels(in_channels), out_channels(out_channels), 
      stride(stride), initializer_(initializer), 
      W_(new Tensor<Dtype>({kernel_height, kernel_width, in_channels, out_channels})),
      b_(new Tensor<Dtype>({out_channels})) {
    InitParams();
  }

  // directly pass in W & b
  Conv2D(size_t kernel_height, size_t kernel_width, size_t in_channels, 
    size_t out_channels, size_t stride, Tensor<Dtype>* W, Tensor<Dtype>* b):
      kernel_height(kernel_height), kernel_width(kernel_width),
      in_channels(in_channels), out_channels(out_channels), 
      stride(stride), W_(W), b_(b) {}

  ~Conv2D() {
    if(W_ != NULL) {
      delete W_;
      W_ = NULL;
    }
    if(b_ != NULL) {
      delete b_;
      b_ = NULL;
    }
  }

  __global__ void conv(Tensor<Dtype> * bottom, Tensor<Dtype> * top, int b, int o) {
    // b is the index of the tensor
    // o is the output channel
    int x_top = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y_top = (blockDim.y * blockIdx.y) + threadIdx.y;
    int x = x_top*stride;
    int y = y_top*stride;
    if (!top->isValidIdx({b, y, x, o})) {
      return;
    }
    std::vector<int> idx = {b, y, x, o};
    size_t in_channels = bottom->GetDims[3];
    Dtype sum = 0.0;
    for(int c = 0; c < in_channels; c++) {
      for(int i = 0; i < kernel_height; i++) {
        for(int j = 0; j < kernel_width; j++) {
          // (n, hei, wid, channel),   // (hei, wid, input, output)
          sum += bottom->atPadding({idx[0], idx[1]+i-int(kernel_height/2), idx[2]+j-int(kernel_width/2), c}) * W_->at({i, j, c, idx[3]});
        }
      }
    }
    sum += b_->at({0});
    top->at({b, y_top, x_top, o}) = sum;
  }

  const int BLOCKDIM = 32;

  void Forward(Tensor<Dtype> * bottom, Tensor<Dtype> * top) {
    // Assert dimensions (n, hei, wid, channel)
    assert(bottom->GetDims()[3]==in_channels);
    assert(top->GetDims()[3]==out_channels);
    assert(bottom->GetDims().size() == 4);
    assert(top->GetDims().size() == 4);
    assert(bottom->GetDims()[0] == top->GetDims()[0]);
    if (Session::GetSession()->gpu) {
      // TODO: implement GPU convolution
      // 1) decide the blocksize and number of blocks
      // 
      size_t n = bottom->GetDims[0];
      size_t hei = top->GetDims[1];
      size_t wid = top->GetDims[2];
      size_t out_channels = top->GetDims[3];

      dim3 blocksInGrid(wid / BLOCKDIM + 1, hei / BLOCKDIM + 1);
      dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
      for (int b = 0; b < n; b++) {
        for (int o = 0; o < out_channels; o++) {
          conv << <blocksInGrid, threadsPerBlock >> > (bottom, top, b, o);
        }
      }
    } else {
      for(int b = 0; b < bottom->GetDims()[0]; b++) {
        for(int o = 0; o < out_channels; o++) {
          for(int x = 0, x_top = 0; x < bottom->GetDims()[2]; x += stride, x_top += 1) {
            for(int y = 0, y_top = 0; y < bottom->GetDims()[1]; y += stride, y_top += 1) {
              // batch idx b, output layer o, pixel (x, y)
              // top->at({b, y, x, o}) = 
              std::vector<int> idx = {b, y, x, o};

              Dtype sum = 0.0;
              for(int c = 0; c < in_channels; c++) {
                for(int i = 0; i < kernel_height; i++) {
                  for(int j = 0; j < kernel_width; j++) {
                    // (n, hei, wid, channel),   // (hei, wid, input, output)
                    sum += bottom->atPadding({idx[0], idx[1]+i-int(kernel_height/2), idx[2]+j-int(kernel_width/2), c}) * W_->at({i, j, c, idx[3]});
                  }
                }
              }
              sum += b_->at({0});
              top->at({b, y_top, x_top, o}) = sum;
              //conv(bottom, top, {b, y, x, o});
            }
          }
        }
      }
    }
  }
  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

  const size_t kernel_height;
  const size_t kernel_width;
  const size_t in_channels;
  const size_t out_channels;
  const size_t stride;

private:
  Tensor<Dtype>* W_;
  Tensor<Dtype>* b_;
  const Initializer<Dtype>* initializer_;
  void InitParams() {
    if (initializer_!=NULL) {
      initializer_->Initialize(W_, b_);
    } else {
      // ConstInitializer<Dtype>((Dtype)0.1, (Dtype)0).Initialize(W_, b_);
      GaussianKernelInitializer<Dtype>((Dtype)5.0).Initialize(W_, b_);
    }
  }

  // // cpu kernel operation
  // void conv(Tensor<Dtype> * bottom, Tensor<Dtype> * top, const std::vector<int> idx) {
  //   // idx = {b, y, x, o}
  //   // batch idx b, output layer o, pixel (x, y)
  //   Dtype sum = 0.0;
  //   for(int c = 0; c < in_channels; c++) {
  //     for(int i = 0; i < kernel_height; i++) {
  //       for(int j = 0; j < kernel_width; j++) {
  //         // (n, hei, wid, channel),   // (hei, wid, input, output)
  //         sum += bottom->atPadding({idx[0], idx[1]+i-int(kernel_height/2), idx[2]+j-int(kernel_width/2), c}) * W_->at({i, j, c, idx[3]});
  //       }
  //     }
  //   }
  //   sum += b_->at({0});
  //   top->at(idx) = sum;
  // }
};

#endif  // CONV2D_LAYER_CUH_
