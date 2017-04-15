#ifndef CONST_INITIALIZER_CUH_
#define CONST_INITIALIZER_CUH_

#include "basics/initializer.hpp"

// Initialize the parameters with a constant value
template<class Dtype>
class ConstInitializer: public Initializer<Dtype> {
public:
  ConstInitializer(Dtype w_val, Dtype b_val): w_val_(w_val), b_val_(b_val) {}
  void Initialize(Tensor<Dtype>* W, Tensor<Dtype>* b) const {
    if (W->gpu) {
      // TODO: GPU initializer
    } else {
      // cpu
      Dtype* w_data_array = W->GetDataPtr();
      for (int i = 0; i < W->size(); i++) {
        w_data_array[i] = w_val_;
      }
      Dtype* b_data_array = b->GetDataPtr();
      for (int i = 0; i < b->size(); i++) {
        b_data_array[i] = b_val_;
      }
    }
  }

private:
  const Dtype w_val_;
  const Dtype b_val_;
};

#endif // CONST_INITIALIZER_CUH_