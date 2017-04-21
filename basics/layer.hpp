#ifndef LAYER_HPP_
#define LAYER_HPP_

#include "tensor.cu"
#include <vector>

// TODO: discuss and finalize interfaces: forward() & backward()

template <class Dtype>
class Layer {
public:
  virtual void Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) = 0;
  // virtual std::vector<Tensor<Dtype>* > Forward(const std::vector<Tensor<Dtype> *> &bottom) = 0;
  // virtual void Backward(Packet& bottom, Packet& top) = 0;
};


#endif  // LAYER_HPP_
