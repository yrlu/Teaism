#ifndef DROPOUT_LAYER_CUH_
#define DROPOUT_LAYER_CUH_

#include "basics/layer.hpp"


template <class Dtype>
class Dropout: public Layer<Dtype> {
public:

  Dropout(size_t seed = NULL);

  ~Dropout() {}

  void Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops);   
  


private:
  curandState* dev_states_;
};

template <class Dtype>
Dropout::Dropout(size_t seed = NULL) {


}

#endif  // DROPOUT_LAYER_CUH_
