#include <iostream>
#include "layers/conv2d.hpp"
#include "basics/tensor.hpp"
#include <assert.h>

void test_conv_layer() {
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Conv2D<float>* conv_layer = new Conv2D<float>(5, 5, 3, 5, 1);
  // std::cout<< conv_layer->kernel_width << " " << conv_layer->kernel_height <<std::endl;
  assert(conv_layer->kernel_height==5);
  assert(conv_layer->kernel_width==5);
  assert(conv_layer->in_channels==3);
  assert(conv_layer->out_channels==5);
  assert(conv_layer->stride==1);
  delete conv_layer;
}

void test_tensor() {
  // inputs: tensor dimensions
  Tensor<float>* tensor = new Tensor<float>({3,3,3});
  // std::cout<< tensor->GetIdx({2,2,2}) <<std::endl;
  assert(tensor->GetIdx({2,2,2})==26);
  // std::cout<< tensor->GetIdx({1,2,2}) <<std::endl;
  assert(tensor->GetIdx({1,2,2})==17);
  // std::cout<< tensor->GetIdx({2,1,2}) <<std::endl;
  assert(tensor->GetIdx({2,1,2})==23);
  // std::cout<< tensor->GetIdx({2,2,1}) <<std::endl;
  assert(tensor->GetIdx({2,2,1})==25);
  delete tensor;
}

int main(void) {
  test_conv_layer();
  test_tensor();  
}