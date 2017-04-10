#ifndef CONST_INITIALIZER_CUH_
#define CONST_INITIALIZER_CUH_

#include "initializer.hpp"

// Initialize the parameters with a constant value
template<class Dtype>
class ConstInitializer: public Initializer<Dtype> {
public:
  ConstInitializer(Dtype val): val_(val) {}
  void Initialize(Tensor<Dtype>& tensor) const {
    Dtype* data_array = tensor.GetDataPtr();
    // cpu
    for (int i = 0; i < tensor.size(); i++) {
      data_array[i] = val_;
    }
    // TODO: GPU initializer
  }

private:
  const Dtype val_;
};

#endif // CONST_INITIALIZER_CUH_