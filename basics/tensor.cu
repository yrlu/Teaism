#ifndef TENSOR_CUH_
#define TENSOR_CUH_

#include <vector>
#include <assert.h>
#include <cstdlib>
#include <numeric>
#include <functional>
#include <iostream>
#include "basics/session.hpp"
#include "cuda_runtime.h"
#include "utils/helper_cuda.h"




template<class Dtype>
class Tensor {
public:
  __device__ Tensor(std::vector<size_t> dims, const bool _gpu=false):dims_(dims), gpu(_gpu) {
    len_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    AllocateMemory();
  }

  __device__ ~Tensor() {
    if (gpu) {
      // TODO: free GPU memory
      if (data_array_ != NULL) {
        cudaFree(data_array_);
        data_array_ = NULL;
      }
    } else {
      // cpu
      if (data_array_ != NULL) {
        free(data_array_);
        data_array_ = NULL;
      }
    }
  }

  __device__ unsigned GetIdx(const std::vector<int> idx) const {
    unsigned out_idx = 0;
    for (int i = 0; i < idx.size(); i++) 
      out_idx = out_idx*dims_[i] + idx[i];
    return out_idx;
  }

  __device__ const std::vector<size_t>& GetDims() const {
    return dims_;
  }

  __device__ Dtype* GetDataPtr() const {
    return data_array_;
  }

  __device__ Dtype& at(const std::vector<int> idx) {
    assert(isValidIdx(idx));
    return data_array_[GetIdx(idx)];
  }

  __device__ const Dtype atPadding(const std::vector<int> idx, Dtype default_val = 0.0) const {
    assert(idx.size() == dims_.size());
    if (!isValidIdx(idx)) return default_val;
    return data_array_[GetIdx(idx)];
  }

  __device__ bool isValidIdx(const std::vector<int> idx) const {
    if(idx.size() != dims_.size()) return false;
    for(int i = 0; i < idx.size(); i++) {
      if(idx[i] < 0 && idx[i] >= dims_.size()) return false;
    }
    return true;
  }

  __device__ size_t size() const {
    return len_;
  }

  const bool gpu;
private:
  std::vector<size_t> dims_;
  size_t len_;
  Dtype* data_array_;
  void AllocateMemory() {
    if (gpu) {
      // TODO: implement GPU memory allocation
      cudaError_t cudaStatus = cudaMalloc((void **)&data_array_, len_*sizeof(Dtype));
      checkCudaErrors(cudaStatus);
    } else {
      // CPU
      data_array_ = (Dtype*)std::malloc(len_*sizeof(Dtype));  
    }
  }
};

#endif // TENSOR_CUH_
