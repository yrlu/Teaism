#ifndef TENSOR_CUH_
#define TENSOR_CUH_

#include <vector>
#include <cstdlib>
#include <numeric>
#include <functional>

template<class Dtype>
class Tensor {
public:
  Tensor(std::vector<size_t> dims):dims_(dims) {
    len_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    AllocateMemory();
  }
  ~Tensor() {
    // cpu
    free(data_array_);
    // TODO: free GPU memory 
  }

  unsigned GetIdx(const std::vector<unsigned> idx) {
    unsigned out_idx = 0;
    for (int i = 0; i < idx.size(); i++) 
      out_idx = out_idx*dims_[i] + idx[i];
    return out_idx;
  }

  Dtype at(const std::vector<unsigned> idx) {
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

private:
  std::vector<size_t> dims_;
  size_t len_;
  Dtype* data_array_;
  void AllocateMemory() {
    // CPU
    data_array_ = (Dtype*)std::malloc(len_*sizeof(Dtype));
    // TODO: implement GPU memory allocation
  }
};

#endif // TENSOR_CUH_
