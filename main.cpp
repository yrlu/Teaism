#include <iostream>
#include "layers/conv2d.hpp"
#include "basics/tensor.hpp"
#include "basics/session.hpp"
#include "tmp/bitmap_image.hpp"
#include "initializers/gaussian_kernel_initializer.hpp"
#include <assert.h>
#include <cmath>

void test_session() {
  std::cout<< "Testing Session .."<<std::endl;
  Session* session = Session::GetNewSession();
  session->gpu = true;
  std::cout<< "use gpu: "<< session->gpu <<std::endl;
}

void test_conv_layer() {
  std::cout<< "Testing Conv2D .."<<std::endl;

  Session* session = Session::GetNewSession();
  session->gpu = false;

  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Conv2D<float>* conv_layer = new Conv2D<float>(5, 5, 1, 1, 1);
  
  const char* OUTPUT_BMP_PATH = "./tmp/test/out.bmp";

  size_t h = 200;
  size_t w = 200;
  bitmap_image img(w, h);
  Tensor<float> bottom = Tensor<float>({1, h, w, 1});

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      bottom.at({0, i, j, 0}) = (float) (rand() % 255);
    }
  }
  // (n, hei, wid, channel)

  Tensor<float> top = Tensor<float>({1, h, w, 1});
  conv_layer->Forward(&bottom, &top);

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      unsigned val = (unsigned) top.at({0, i, j, 0});
      img.set_pixel(j, i, val, val, val);
    }
  }
  
  img.save_image(OUTPUT_BMP_PATH);
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
  for (int i = 0; i < W.GetDims()[0]; i++) {
    for (int j = 0; j < W.GetDims()[1]; j++) {
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