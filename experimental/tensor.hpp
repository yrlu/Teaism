#ifndef TENSOR_CUH_
#define TENSOR_CUH_

#include <vector>

template<class Dtype>
class Tensor {
public:
  Tensor(std::vector<size_t> dims):dims_(dims) {}
  ~Tensor() {}

  unsigned get_idx(const std::vector<unsigned> idx) {
    unsigned out_idx = 0;
    for (int i = idx.size()-1; i >= 0; --i)
      out_idx = out_idx*dims_[i] + idx[i];
    return out_idx;
  }

  Dtype at(const std::vector<unsigned> idx) {
    return data_array_[get_idx(idx)];
  }

private:
  std::vector<size_t> dims_;
  Dtype* data_array_;
};

#endif // TENSOR_CUH_
