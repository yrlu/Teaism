#ifndef NETWORK_CUH_
#define NETWORK_CUH_

#include <assert.h>
#include <cstdlib>
#include "basics/layer.hpp"
#include "basics/session.hpp"
#include "basics/tensor.cu"
#include "cuda_runtime.h"
#include "utils/helper_cuda.h"
#include "stdio.h"

template<class Dtype>
struct LayerData { 
  std::vector<Tensor<Dtype>*> tops; 
  std::vector<Tensor<Dtype>*> tops_diff; 
};

/* 
Network
*/
template<class Dtype>
class Network {

public:
  Network(std::vector<Layer<Dtype>*> layers):layers_(layers) {
    InitNetwork();
  }

  void Forward();

  void Backward();

  void Update();

  void InitNetwork();
  
  LayerData<Dtype>* GetLayerData(int idx) {}

  std::vector<Layer<Dtype>*> layers;

private:
  std::vector<std::pair<Layer<Dtype>*, LayerData<Dtype>*>> layer_data_pairs_;
  std::vector<Layer<Dtype>*> & layers_;
  unsigned num_layers_;
};

template<class Dtype>
void Network<Dtype>::Forward() {
  // for(unsigned i = 0; i < num_layers; ++i) {
    // layer_data_pairs[i].first->Forward
}

template<class Dtype>
void Network<Dtype>::Backward() {

}

template<class Dtype>
void Network<Dtype>::Update() {

}

template<class Dtype>
void Network<Dtype>::InitNetwork() {

}
#endif // NRTWORK_CUH_
