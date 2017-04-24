
#include <stdio.h>
#include <string>
#include <sstream>
#include <iostream>
#include "basics/tensor.cu"
#include "layers/conv2d.cu"
#include "layers/fc.cu"
#include "utils/load_model.hpp"

void show_tensor(Tensor<float>* t) {
  const size_t* dims = t->GetDims();
  for (int o = 0; o < dims[3]; ++o) {
    for (int i = 0; i < dims[2]; ++i) {
      for (int w = 0; w < dims[1]; ++w) {
        for (int h = 0; h < dims[0]; ++h) {
          printf("(%d, %d, %d, %d): %f\n", h, w, i, o, t->at(h,w,i,o));
        }
      }
    }
  }
}

void show_fc_data(Tensor<float>* t) {
  const size_t* dims = t->GetDims();
  for (int o = 0; o < dims[2]; ++o) {
    for (int i = 0; i < dims[3]; ++i) {
      printf("(%d, %d, %d, %d): %f\n", 0,0,o,i, t->at(0,0,o,i));
    }
  }
}

void test_load_model() {

  std::string model_path = "models/alexnet/model.txt";

  std::ifstream file(model_path);
  
  // conv1_w
  size_t conv1_w_dims[4] = {11,11,3,96};
  Tensor<float>* conv1_w = Tensor<float>::CreateTensorCPU(conv1_w_dims);
//  load_to_conv<float>(conv1_w, file);

  // fc8_w
  size_t fc8_w_dims[4] = {1,1,1000,4096};
  Tensor<float>* fc8_w = Tensor<float>::CreateTensorCPU(fc8_w_dims);
  load_to_fc(fc8_w, file);

  // fc8_b
  size_t fc8_b_dims[4] = {1,1,1,1000};
  Tensor<float>* fc8_b = Tensor<float>::CreateTensorCPU(fc8_b_dims);
  load_to_bias(fc8_b, file);

//  show_tensor(conv1_w);
//  show_fc_data(fc8_w);
//  show_tensor(fc8_b);
}


int main() {
  test_load_model();
}
