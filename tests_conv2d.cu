#include <stdio.h>
#include "basics/tensor.cu"
#include "initializers/gaussian_kernel_initializer.cu"
#include <assert.h>
#include <cmath>
#include "basics/session.hpp"
#include "layers/conv2d.cu"
#include "tmp/bitmap_image.hpp"


void test_conv2d_cpu() {
  printf("Example code for conv2d cpu\n");
  size_t h = 400;
  size_t w = 600;

  Session* session = Session::GetNewSession();
  session->gpu = false;
 
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Conv2D<float> * conv_layer = new Conv2D<float>(5,5,1,1,2);
  const char* OUTPUT_BMP_PATH = "./tmp/test/out.bmp";

  size_t b_dims[4] = {1, h, w, 1};
  Tensor<float>* bottom = Tensor<float>::CreateTensorCPU(b_dims);
  size_t t_dims[4] = {1, h/2, w/2, 1};
  Tensor<float>* top = Tensor<float>::CreateTensorCPU(t_dims);

  for(int i = 0; i < h; i++) {
  	for(int j = 0; j < w; j++) {
  	  int b_idx[4] = {0, i, j, 0};
  	  bottom->at(b_idx) = (float) (rand() % 255);
  	}
  }
  conv_layer->Forward(bottom, top);

  bitmap_image img(w/2, h/2);
  for (int i = 0; i < h/2; i++) {
    for (int j = 0; j < w/2; j++) {
      unsigned val = (unsigned) top->at(0, i, j, 0);
      img.set_pixel(j, i, val, val, val);
    }  
  }	
  img.save_image(OUTPUT_BMP_PATH);
  delete conv_layer;
}


__global__ void init_bottom(Tensor<float> * bottom) {
  for(int i = 0; i < bottom->GetDims()[1]; i++) {
  	for(int j = 0; j < bottom->GetDims()[2]; j++) {
  	  int b_idx[4] = {0, i, j, 0};
  	  bottom->at(b_idx) = (float) ((i+j) % 255);
  	}
  }
}

__global__ void show_top(Tensor<float>* top) {
  size_t h = top->GetDims()[1];
  size_t w = top->GetDims()[2];
  for (int i = 0; i < h/2; i++) {
    for (int j = 0; j < w/2; j++) {
  	  printf("%f ", top->at(0, i, j, 0));
    }
    printf("\n");
  }	
  printf("%d \n", top->GetDataPtr());

}

void test_conv2d_gpu() {
  printf("Example code for conv2d gpu\n");
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  size_t h = 400;
  size_t w = 600;

  Session* session = Session::GetNewSession();
  session->gpu = true;
 
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Conv2D<float> * conv_layer = new Conv2D<float>(5,5,1,1,2);
  const char* OUTPUT_BMP_PATH = "./tmp/test/out_gpu.bmp";

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  size_t b_dims[4] = {1, h, w, 1};
  Tensor<float>* bottom = Tensor<float>::CreateTensorGPU(b_dims);
  
  size_t t_dims[4] = {1, h/2, w/2, 1};
  Tensor<float>* top = Tensor<float>::CreateTensorGPU(t_dims);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  init_bottom<<<1,1>>>(bottom);
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  conv_layer->Forward(bottom, top);
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  show_top<<<1,1>>>(top);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
 
  Tensor<float> * top_cpu = Tensor<float>::TensorGPUtoCPU(top);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  // printf("%d %d %d %d\n", top_cpu->GetDims()[0], top_cpu->GetDims()[1], top_cpu->GetDims()[2], top_cpu->GetDims()[3]);
  // printf("%d \n", top_cpu->size());
  // printf("%d \n", top_cpu->GetDataPtr());
  // bitmap_image img(w/2, h/2);	
  // for (int i = 0; i < h/2; i++) {
  //   for (int j = 0; j < w/2; j++) {
  //     unsigned val = (unsigned) top_cpu->at(0, i, j, 0);
  //     img.set_pixel(j, i, val, val, val);
  //   }
  // }
  // img.save_image(OUTPUT_BMP_PATH);
  // delete conv_layer;
  // delete top_cpu;
  // cudaFree(bottom);
  // cudaFree(top);
}


int main() {
  test_conv2d_cpu();
  test_conv2d_gpu();
}