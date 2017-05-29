#ifndef NETWORK_CUH_
#define NETWORK_CUH_

#include <assert.h>
#include <cstdlib>
#include "basics/layer.hpp"
#include "basics/session.hpp"
#include "basics/tensor.hpp"
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

  void Forward();

  void Backward();

  void Update();
  
  LayerData<Dtype>* GetLayerData(int idx) {
    return layer_data_pairs[idx].second; }

  std::vector<Layer*> layers;

private:
  std::vector<std::pair<Layer*, LayerData<Dtype>*>> layer_data_pairs_;
  unsigned num_layers_;
};

template<class Dtype>
void Network::Forward() {
  for(unsigned i = 0; i < num_layers; ++i) {
    layer_data_pairs[i].first->Forward
}

template<class Dtype>
void Network::Backward() {

}

template<class Dtype>
void Network::Update() {

}
#endif // NRTWORK_CUH_
