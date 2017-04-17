#ifndef INITIALIZER_CUH_
#define INITIALIZER_CUH_

#include "tensor.cu"

template<class Dtype>
class Initializer {
public:
  __device__ virtual void Initialize(Tensor<Dtype>* W, Tensor<Dtype>* b) const = 0;
};

#endif // INITIALIZER_CUH_