#ifndef INITIALIZER_CUH_
#define INITIALIZER_CUH_

#include "tensor.cu"

template<class Dtype>
class Initializer {
public:
  virtual void Initialize(Tensor<Dtype>* W, Tensor<Dtype>* b, bool gpu = true) const = 0;
};

#endif // INITIALIZER_CUH_