#ifndef NETWORK_CUH_
#define NETWORK_CUH_

#include <assert.h>
#include <cstdlib>
#include "basics/layer.hpp"
#include "basics/tensor.cu"

template<class Dtype>
class LayerData {
public:
  std::vector<Tensor<Dtype>*> tops;
  std::vector<Tensor<Dtype>*> tops_diff;
};

/* 
Network
*/
template<class Dtype>
class Network {
public:
  Network(std::vector<Layer<Dtype>*> layers):layers(layers) {
    InitNetwork();
  }

  void InitNetwork();

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

template<class Dtype>
void Network<Dtype>::InitNetwork() {

  Session * session = Session::GetSession();
  if(layers.size() == 0) return;
  // special handling for data layer
  size_t bottom_dims[4];
  size_t data_tops_dims1[4];

  layers[0]->GetTopsDims({}, {bottom_dims, data_tops_dims1});

  Tensor<Dtype> * data_top0;
  Tensor<Dtype> * data_top1;
  Tensor<Dtype> * data_top_diff0;

  if(session->gpu == true) {
    data_top0 = Tensor<Dtype>::CreateTensorGPU(bottom_dims);
    data_top1 = Tensor<Dtype>::CreateTensorGPU(data_tops_dims1);
    data_top_diff0 = Tensor<Dtype>::CreateTensorGPU(bottom_dims);
  } else {
    data_top0 = Tensor<Dtype>::CreateTensorCPU(bottom_dims);
    data_top1 = Tensor<Dtype>::CreateTensorCPU(data_tops_dims1);
    data_top_diff0 = Tensor<Dtype>::CreateTensorCPU(bottom_dims);
  }
  LayerData<Dtype>* ld = new LayerData<Dtype>();
  ld->tops = {data_top0, data_top1};
  ld->tops_diff = {data_top_diff0};

  layer_data_pairs_.push_back(std::make_pair(layers[0], ld));
  // common layers
  for(int i = 1; i < layers.size(); i++) {
    Layer<Dtype> * cur_layer = layers[i];
    size_t top_dims[4];
    cur_layer->GetTopsDims({bottom_dims}, {top_dims});
    for(int j = 0; j < 4; j++)
      bottom_dims[j] = top_dims[j];
    Tensor<Dtype> * top;
    Tensor<Dtype> * top_diff;
    if (session->gpu == true) {
      top = Tensor<Dtype>::CreateTensorGPU(top_dims);
      top_diff = Tensor<Dtype>::CreateTensorGPU(top_dims);
    } else {
      top = Tensor<Dtype>::CreateTensorCPU(top_dims);
      top_diff = Tensor<Dtype>::CreateTensorCPU(top_dims);
    }

    LayerData<Dtype>* ld = new LayerData<Dtype>();
    ld->tops = {top};
    ld->tops_diff = {top_diff};
    layer_data_pairs_.push_back(std::make_pair(cur_layer, ld));
  }

}



#endif // NRTWORK_CUH_
