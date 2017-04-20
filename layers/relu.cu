#ifndef RELU_LAYER_CUH_
#define RELU_LAYER_CUH_

#include "basics/layer.hpp"
#include "basics/tensor.cu"
#include "basics/session.hpp"

#define BLOCKDIM 32

template <class Dtype>
__global__ void relu(Tensor<Dtype> * bottom, Tensor<Dtype> * top, int bi, int o) {
  // bi is the index of the tensor
  // o is the output channel
  int x_top = (blockDim.x * blockIdx.x) + threadIdx.x;
  int y_top = (blockDim.y * blockIdx.y) + threadIdx.y;
  int x = x_top;
  int y = y_top;
  if (!bottom->isValidIdx(bi, y, x, o) || !top->isValidIdx(bi, y_top, x_top, o)) {
    return;
  }
  Dtype val = bottom->at(bi, y, x, o);
  top->at(bi, y_top, x_top, o) = (val >= 0 ? val : 0);
}

template <class Dtype>
__global__ void ReluForwardGPU(Tensor<Dtype> * bottom, Tensor<Dtype> * top) {
  size_t n = bottom->GetDims()[0];
  size_t hei = top->GetDims()[1];
  size_t wid = top->GetDims()[2];
  size_t out_channels = top->GetDims()[3];

  dim3 blocksInGrid(wid / BLOCKDIM + 1, hei / BLOCKDIM + 1);
  dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
  for (int b = 0; b < n; b++) {
    for (int o = 0; o < out_channels; o++) {
      relu<Dtype><<<blocksInGrid, threadsPerBlock>>>(bottom, top, b, o);
    }
  }
}


template <class Dtype>
class Relu: public Layer<Dtype> {
public:
  void Forward(Tensor<Dtype> * bottom, Tensor<Dtype> * top) {
    if (Session::GetSession()->gpu) {
      ReluForwardGPU<<<1,1>>>(bottom, top);
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
};

#endif 