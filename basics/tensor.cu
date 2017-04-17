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
  __host__ __device__ Tensor(size_t* dims, const bool _gpu=true):gpu(_gpu) {
    //len_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    len_ = dims[0] * dims[1] * dims[2] * dims[3];
    dims_[0] = dims[0];
    dims_[1] = dims[1];
    dims_[2] = dims[2];
    dims_[3] = dims[3];
  }

  __host__ __device__ ~Tensor() {
      // TODO: free GPU memory
  }

  __host__ __device__ unsigned GetIdx(const int* idx) const {
    unsigned out_idx = 0;
    for (int i = 0; i < 4; i++)
      out_idx = out_idx*dims_[i] + idx[i];
    return out_idx;
  }

  __host__ __device__ const size_t* GetDims() const {
    return dims_;
  }

  __host__ __device__ Dtype* GetDataPtr() const {
    return data_array_;
  }

  __host__ __device__ Dtype& at(const int* idx) {
    assert(isValidIdx(idx));
    return data_array_[GetIdx(idx)];
  }

  __host__ __device__ const Dtype atPadding(int* idx, Dtype default_val = 0.0) const {
    if (!isValidIdx(idx)) return default_val;
    return data_array_[GetIdx(idx)];
  }

  __host__ __device__ bool isValidIdx(const int* idx) const {
    for(int i = 0; i < 4; i++) {
      if(idx[i] < 0 && idx[i] >= 4) return false;
    }
    return true;
  }

  __host__ __device__ size_t size() const {
    return len_;
  }

  const bool gpu;
  size_t dims_[4];
  size_t len_;
  Dtype* data_array_;

private:
};

#endif // TENSOR_CUH_
