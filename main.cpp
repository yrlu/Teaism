#include <iostream>
#include "layers/conv2d.hpp"
#include "basics/tensor.hpp"
#include "basics/session.hpp"
#include "initializers/gaussian_kernel_initializer.hpp"
#include <assert.h>
#include <cmath>

void test_session() {
  std::cout<< "Testing Session .."<<std::endl;
  Session* session = Session::GetSession();
  session->gpu = true;
  std::cout<< "use gpu: "<< session->gpu <<std::endl;
}

void test_conv_layer() {
  std::cout<< "Testing Conv2D .."<<std::endl;
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Conv2D<float>* conv_layer = new Conv2D<float>(5, 5, 3, 5, 1);
  assert(conv_layer->kernel_height==5);
  assert(conv_layer->kernel_width==5);
  assert(conv_layer->in_channels==3);
  assert(conv_layer->out_channels==5);
  assert(conv_layer->stride==1);
  delete conv_layer;
}

void test_tensor() {
  std::cout<< "Testing Tensor .."<<std::endl;
  // inputs: tensor dimensions
  Tensor<float>* tensor = new Tensor<float>({3,3,3});
  assert(tensor->GetIdx({2,2,2})==26);
  assert(tensor->GetIdx({1,2,2})==17);
  assert(tensor->GetIdx({2,1,2})==23);
  assert(tensor->GetIdx({2,2,1})==25);
  delete tensor;
}

void test_gaussian_kernel() {
  std::cout<< "Testing gaussian kernel initializer .. "<<std::endl;  
  std::cout<<Session::GetNewSession()->gpu<<std::endl;
  Tensor<float>W = Tensor<float>({5,5,1,1});
  Tensor<float>b = Tensor<float>({1});
  GaussianKernelInitializer<float>(5.0).Initialize(&W, &b);
  double sum = 0.0;
  for (unsigned i = 0; i < W.GetDims()[0]; i++) {
    for (unsigned j = 0; j < W.GetDims()[1]; j++) {
      sum += W.at({i, j, 0, 0});
      std::cout<<W.at({i, j, 0, 0})<<"\t";
    }
    std::cout<<std::endl;
  }
  assert(std::abs(sum-1.0)<0.00001);
}


int main(void) {
  test_conv_layer();
  test_tensor();  
  test_session();
  test_gaussian_kernel();
}