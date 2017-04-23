
#ifndef LRN_LAYER_CUH_
#define LRN_LAYER_CUH_

#include <assert.h>
#include <cmath>
#include "basics/layer.hpp"
#include "basics/tensor.cu"
#include "basics/session.hpp"

// TODO: implement CUDA kernel for backward()

#define BLOCKDIM 32

namespace LRNGPUKernels {

  template <class Dtype>
  __global__ void ForwardGPUKernel(
    Tensor<Dtype>* bottom, Tensor<Dtype>* top, int bi, int local_size, float alpha, float beta, int extend_) {
    int h = (blockDim.x * blockIdx.x) + threadIdx.x;
    int w = (blockDim.y * blockIdx.y) + threadIdx.y;
    int nchannels = bottom->GetDims()[3];

    if (!top->isValidIdx(bi,h,w,0)) { return; }

    for (int ch = 0; ch < nchannels; ++ch) {
      top->at(bi,h,w,ch) = 0;
    } 

    for (int ch = 0; ch < nchannels; ++ch) {
      for (int l = -extend_; l <= extend_; ++l) {
        if (ch+l >= 0 && ch+l < nchannels) {
          top->at(bi,h,w,ch+l) += pow(bottom->at(bi,h,w,ch), 2); } } }

    for (int ch = 0; ch < nchannels; ++ch) {
      int num_n = (ch<extend_)? (ch+extend_+1) :
                  ((ch>=nchannels-extend_)? (ch-nchannels+extend_+2) : (local_size));
      top->at(bi,h,w,ch) = pow(1+alpha/num_n*top->at(bi,h,w,ch), beta); } 
  }

  template <class Dtype>
  __global__ void ForwardGPU(
    Tensor<Dtype>* bottom, Tensor<Dtype>* top, int local_size, float alpha, float beta, int extend_) {
    assert(bottom->GetDims()[0] == top->GetDims()[0]);  // bottom channel should be equal to top channel
    assert(bottom->GetDims()[1] == top->GetDims()[1]);
    assert(bottom->GetDims()[2] == top->GetDims()[2]);
    assert(bottom->GetDims()[3] == top->GetDims()[3]);

    int batch_idx = threadIdx.x;
    dim3 blocksInGrid(bottom->GetDims()[1]/BLOCKDIM+1, bottom->GetDims()[2]/BLOCKDIM+1);
    dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
    LRNGPUKernels::ForwardGPUKernel<Dtype><<<blocksInGrid,threadsPerBlock>>>
      (bottom, top, batch_idx, local_size, alpha, beta, extend_);
  }

}

template <class Dtype>
class LRN: public Layer<Dtype> {
public:
  LRN(int local_size = 5, float alpha = 0.0001, float beta = 0.75):
      local_size(local_size), alpha(alpha), beta(beta), extend_((local_size-1)/2) {};

  ~LRN() {}

  void Forward(const std::vector<Tensor<Dtype>*> &, const std::vector<Tensor<Dtype>*> &);

  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

  void GetTopsDims(const std::vector<size_t*> &, const std::vector<size_t*> &); 

  int local_size;
  float alpha;
  float beta;

private:
  int extend_;
};

template <class Dtype>
void LRN<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) {
  assert(bottoms.size() == 1);  // Need only one bottom tensor
  assert(tops.size() == 1);  // Need only one bottom tensor

  Session* S = Session::GetSession();
  if (S->gpu) {
    LRNGPUKernels::ForwardGPU<Dtype><<<1, S->batch_size>>>(bottoms[0], tops[0], local_size, alpha, beta, extend_);
  } else {
    assert(bottoms[0]->GetDims()[0] == tops[0]->GetDims()[0]);  // bottom dims should be equal to top dims
    assert(bottoms[0]->GetDims()[1] == tops[0]->GetDims()[1]);
    assert(bottoms[0]->GetDims()[2] == tops[0]->GetDims()[2]);
    assert(bottoms[0]->GetDims()[3] == tops[0]->GetDims()[3]);

    const size_t batch_size = bottoms[0]->GetDims()[0];
    const size_t nchannels = bottoms[0]->GetDims()[3];

    for (int bi = 0; bi < batch_size; ++bi) {
      for (int i = 0; i < bottoms[0]->GetDims()[1]; ++i) {
        for (int j = 0; j < bottoms[0]->GetDims()[2]; ++j) {
          for (int ch = 0; ch < nchannels; ++ch) {
            tops[0]->at(bi,i,j,ch) = 0; } } } }

    for (int bi = 0; bi < batch_size; ++bi) {
      for (int i = 0; i < bottoms[0]->GetDims()[1]; ++i) {
        for (int j = 0; j < bottoms[0]->GetDims()[2]; ++j) {
          for (int ch = 0; ch < nchannels; ++ch) {
            for (int l = -extend_; l <= extend_; ++l) {
              if (ch+l >= 0 && ch+l < nchannels) {
                tops[0]->at(bi,i,j,ch+l) += pow(bottoms[0]->at(bi,i,j,ch), 2); } } } } } }

    for (int bi = 0; bi < batch_size; ++bi) {
      for (int i = 0; i < bottoms[0]->GetDims()[1]; ++i) {
        for (int j = 0; j < bottoms[0]->GetDims()[2]; ++j) {
          for (int ch = 0; ch < nchannels; ++ch) {
            int num_n = (ch<extend_)? (ch+extend_+1) : 
                        ((ch>=nchannels-extend_)? (ch-nchannels+extend_+2) : (local_size));
            tops[0]->at(bi,i,j,ch) = pow(1+alpha/num_n*tops[0]->at(bi,i,j,ch), beta); } } } }
  }
}

template <class Dtype>
void LRN<Dtype>::GetTopsDims(const std::vector<size_t*> &bottoms_dims, const std::vector<size_t*> &tops_dims) {
  assert(bottoms_dims.size() == 1);
  assert(tops_dims.size() == 1);

  tops_dims[0][0] = bottoms_dims[0][0];
  tops_dims[0][1] = bottoms_dims[0][1];
  tops_dims[0][2] = bottoms_dims[0][2];
  tops_dims[0][3] = bottoms_dims[0][3];
}


#endif  // SOFTMAX_LAYER_CUH_
