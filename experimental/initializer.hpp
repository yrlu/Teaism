#ifndef INITIALIZER_CUH_
#define INITIALIZER_CUH_

#include "tensor.hpp"

template<class Dtype>
class Initializer {
public:
  virtual void Initialize(Tensor<Dtype>& tensor) const = 0;
};

#endif // INITIALIZER_CUH_