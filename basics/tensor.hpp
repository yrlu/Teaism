#ifndef TENSOR_CUH_
#define TENSOR_CUH_

#include <vector>
#include <assert.h>
#include <cstdlib>
#include <numeric>
#include <functional>
#include <iostream>
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

  unsigned GetIdx(const std::vector<int> idx) const {
    unsigned out_idx = 0;
    for (int i = 0; i < idx.size(); i++) 
      out_idx = out_idx*dims_[i] + idx[i];
    return out_idx;
  }

  const std::vector<size_t>& GetDims() const {
    return dims_;
  }

  Dtype* GetDataPtr() const {
    return data_array_;
  }

  Dtype& at(const std::vector<int> idx) {
    assert(isValidIdx(idx));
    return data_array_[GetIdx(idx)];
  }

  const Dtype atPadding(const std::vector<int> idx, Dtype default_val = 0.0) const {
    assert(idx.size() == dims_.size());
    if (!isValidIdx(idx)) return default_val;
    return data_array_[GetIdx(idx)];
  }

  bool isValidIdx(const std::vector<int> idx) const {
    if(idx.size() != dims_.size()) return false;
    for(int i = 0; i < idx.size(); i++) {
      if(idx[i] < 0 && idx[i] >= dims_.size()) return false;
    }
    return true;
  }

  size_t size() const {
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
    } else {
      // CPU
      data_array_ = (Dtype*)std::malloc(len_*sizeof(Dtype));  
    }
  }
};

#endif // TENSOR_CUH_
