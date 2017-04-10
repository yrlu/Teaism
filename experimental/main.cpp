#include <iostream>
#include "conv2d.hpp"
#include "tensor.hpp"
#include <assert.h>

using namespace std;

void test_conv_layer() {
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Conv2D<float>* conv_layer = new Conv2D<float>(5, 5, 3, 5, 1);
  cout<< conv_layer->kernel_width << " " << conv_layer->kernel_height <<endl;
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
  cout<< tensor->get_idx({2,2,2}) <<endl;
  assert(tensor->get_idx({2,2,2})==26);
  cout<< tensor->get_idx({1,2,2}) <<endl;
  assert(tensor->get_idx({1,2,2})==25);
  cout<< tensor->get_idx({2,1,2}) <<endl;
  assert(tensor->get_idx({2,1,2})==23);
  cout<< tensor->get_idx({2,2,1}) <<endl;
  assert(tensor->get_idx({2,2,1})==17);
  delete tensor;
}

int main(void) {
  test_conv_layer();
  test_tensor();  
  
}