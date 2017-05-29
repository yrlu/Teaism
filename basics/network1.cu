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
  // layer_data_pairs_.push_back(std::make_pair(layers[0], ld));
  // std::vector<std::pair<Layer<Dtype>*, LayerData<Dtype>*>> layer_data_pairs_;
  layer_data_pairs_.push_back(std::make_pair(layers[0], ld));
  // common layers
  for(int i = 1; i < layers.size(); i++) {
    Layer<Dtype> * cur_layer = layers[i];
    size_t top_dims[4];
    cur_layer->GetTopsDims({bottom_dims}, {top_dims});
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
