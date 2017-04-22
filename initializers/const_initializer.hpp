#ifndef CONST_INITIALIZER_CUH_
#define CONST_INITIALIZER_CUH_

#include "basics/initializer.hpp"

// Initialize the parameters with a constant value
template<class Dtype>
class ConstInitializer: public Initializer<Dtype> {
public:
  ConstInitializer(Dtype w_val, Dtype b_val): w_val_(w_val), b_val_(b_val) {}
  void Initialize(Tensor<Dtype>* W, Tensor<Dtype>* b, bool gpu = true) const;
  __host__ __device__ static void InitConst(Tensor<Dtype> * W, Tensor<Dtype> *b, const Dtype w_val, const Dtype b_val) {
    Dtype* w_data_array = W->GetDataPtr();
    const size_t w_len = W->size();
    for(int i = 0; i < w_len; i++) {
      w_data_array[i] = w_val;
    }

    Dtype* b_data_array = b->GetDataPtr();
    const size_t b_len = b->size();
    for(int i = 0; i < b_len; i++) {
      b_data_array[i] = b_val;
    }
  }

private:
  const Dtype w_val_;
  const Dtype b_val_;
};

template <class Dtype>
__global__ void InitializeGPU(Tensor<Dtype> * W, Tensor<Dtype> *b, const Dtype w_val, const Dtype b_val) {
  ConstInitializer<Dtype>::InitConst(W, b, w_val, b_val);
}

template <class Dtype>
void ConstInitializer<Dtype>::Initialize(Tensor<Dtype>* W, Tensor<Dtype>* b, bool gpu) const {
  if (gpu) {
    InitializeGPU<<<1, 1>>>(W, b, sigma_);
  } else {
    ConstInitializer<Dtype>::InitConst(W, b, sigma_);
  }
}


#endif // CONST_INITIALIZER_CUH_