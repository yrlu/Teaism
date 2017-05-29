#ifndef NETWORK_CUH_
#define NETWORK_CUH_

#include <assert.h>
#include <cstdlib>
#include "basics/layer.hpp"
#include "basics/tensor.cu"

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
  
  LayerData<Dtype>* GetLayerData(int idx) {
    return layer_data_pairs_[idx].second; }

  std::vector<Layer<Dtype>*> layers;

private:
  std::vector<std::pair<Layer<Dtype>*, LayerData<Dtype>*>> layer_data_pairs_;
};

template<class Dtype>
void Network<Dtype>::Forward() {
  for (unsigned i = 0; i < layers.size(); ++i) {
    if (i == 0) {  // Data Layer
      layer_data_pairs_[i].first->Forward(
        std::vector<Tensor<Dtype>*> (), layer_data_pairs_[i].second->tops);
    } else if (i == 1) {  // Conv1 Layer
      layer_data_pairs_[i].first->Forward(
        {layer_data_pairs_[i-1].second->tops[0]}, layer_data_pairs_[i].second->tops);
    } else if (i == layers.size() - 1) {  // Loss Layer
      layer_data_pairs_[i].first->Forward(
        {layer_data_pairs_[i-1].second->tops[0], layer_data_pairs_[0].second->tops[1]}, 
        layer_data_pairs_[i].second->tops);        
    } else {
      layer_data_pairs_[i].first->Forward(
        layer_data_pairs_[i-1].second->tops, layer_data_pairs_[i].second->tops);
    }
  }
}

template<class Dtype>
void Network<Dtype>::Backward() {
  for (unsigned i = layers.size() - 1; i > 0; --i) {
    if (i == layers.size() - 1) {  // Loss Layer
      layer_data_pairs_[i].first->Backward(
        layer_data_pairs_[i].second->tops, layer_data_pairs_[i].second->tops,
        {layer_data_pairs_[i-1].second->tops[0], layer_data_pairs_[0].second->tops[1]},
        {layer_data_pairs_[i-1].second->tops_diff[0], layer_data_pairs_[0].second->tops_diff[1]});
    } else if (i == 1) {  // Conv1 Layer
      layer_data_pairs_[i].first->Backward(
        layer_data_pairs_[i].second->tops, layer_data_pairs_[i].second->tops_diff,
        {layer_data_pairs_[i-1].second->tops[0]}, {layer_data_pairs_[i-1].second->tops_diff[0]});
    } else {
      layer_data_pairs_[i].first->Backward(
        layer_data_pairs_[i].second->tops, layer_data_pairs_[i].second->tops_diff,
        layer_data_pairs_[i-1].second->tops_diff, layer_data_pairs_[i-1].second->tops_diff);
    }
  }
}

#endif // NRTWORK_CUH_
