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
  assert(conv_layer->kernel_height==5);
  assert(conv_layer->kernel_width==5);
  assert(conv_layer->in_channels==1);
  assert(conv_layer->out_channels==1);
  assert(conv_layer->stride==1);

  const char* INPUT_BMP_PATH = "./tmp/test/steel_wool_small.bmp";
  const char* OUTPUT_REFERENCE_BMP_PATH = "./tmp/test/steel_wool_large_reference_output.bmp";
  const char* OUTPUT_BMP_PATH = "./tmp/test/out.bmp";

  size_t h = 10;
  size_t w = 10;
  bitmap_image img(w, h);
  Tensor<float> bottom = Tensor<float>({1, h, w, 1});

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      // bottom.at({0, i, j, 0}) = (float)img.red_channel(j, i);
      bottom.at({0, i, j, 0}) = (float) (rand() % 255);
      std::cout<< bottom.at({0, i, j, 0}) << " ";
    }
    std::cout<<std::endl;
  }

  // (n, hei, wid, channel)

  Tensor<float> top = Tensor<float>({1, h, w, 1});
  conv_layer->Forward(&bottom, &top);

  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      unsigned val = (unsigned) top.at({0, i, j, 0});
      std::cout<< top.at({0, i, j, 0}) << " ";
      img.set_pixel(j, i, val, val, val);
    }
    std::cout<<std::endl;
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