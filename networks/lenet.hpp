#ifndef LENET_HPP_
#define LENET_HPP_

#include "basics/tensor.cu"
#include "basics/layer.cu"


class LeNet {
public:
  LeNet(Layer<float>* data_source):data_source_(data_source) {
  	InitializeLayers();
  }

  void Forward() {

  }

  vector<Tensor*> Forward(const vector<Tensor*> img) {

  }

private:
  Layer<float> * data_source_;

  void InitializeLayers() {
  	
  }
}

#endif