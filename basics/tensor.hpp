#ifndef TENSOR_CUH_
#define TENSOR_CUH_

#include <vector>
#include <cstdlib>
#include <numeric>
#include <functional>
#include "basics/session.hpp"

template<class Dtype>
class Tensor {
public:
  Tensor(std::vector<size_t> dims, const bool _gpu=false):dims_(dims), gpu(_gpu) {
    len_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    AllocateMemory();
  }

  ~Tensor() {
    if (gpu) {
      // TODO: free GPU memory 
    } else {
      // cpu
      if (data_array_ != NULL) {
        free(data_array_);
        data_array_ = NULL;
      }
    }
  }

  unsigned GetIdx(const std::vector<unsigned> idx) {
    unsigned out_idx = 0;
    for (int i = 0; i < idx.size(); i++) 
      out_idx = out_idx*dims_[i] + idx[i];
    return out_idx;
  }

  Dtype& at(const std::vector<unsigned> idx) {
    return data_array_[GetIdx(idx)];
  }

  Dtype* GetDataPtr() {
    return data_array_;
  }

  size_t size() {
    return len_;
  }

  const std::vector<size_t>& GetDims() {
    return dims_;
  }

  const bool gpu;
private:
  std::vector<size_t> dims_;
  size_t len_;
  Dtype* data_array_;
  void AllocateMemory() {
    if (gpu) {
      // TODO: implement GPU memory allocation
    } else {
      // CPU
      data_array_ = (Dtype*)std::malloc(len_*sizeof(Dtype));  
    }
  }
};

#endif // TENSOR_CUH_
