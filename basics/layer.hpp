#ifndef LAYER_HPP_
#define LAYER_HPP_

#include "tensor.hpp"

// TODO: discuss and finalize interfaces: forward() & backward()

template <class Dtype>
class Layer {
public:
  virtual void Forward(Tensor<Dtype>& bottom, Tensor<Dtype>& top) = 0;
  // virtual void Backward(Packet& bottom, Packet& top) = 0;
};


#endif  // LAYER_HPP_