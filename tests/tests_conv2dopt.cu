#include <stdio.h>
#include "basics/tensor.cu"
#include "initializers/gaussian_kernel_initializer.cu"
#include <assert.h>
#include <cmath>
#include <vector>
#include "basics/session.hpp"
#include "layers/conv2d.cu"
#include "utils/utils.cu"
#include "utils/bitmap_image.hpp"



__global__ void init_bottom(Tensor<float> * bottom) {
  for(int b = 0; b < bottom->GetDims()[0]; b++) {
    for(int c = 0; c < bottom->GetDims()[1]; c++) {
      for(int i = 0; i < bottom->GetDims()[2]; i++) {
        for(int j = 0; j < bottom->GetDims()[3]; j++) {
          int b_idx[4] = {b, c, i, j};
          bottom->at(b_idx) = (float) ((i+j+c) % 255);
        }
      }
    }
  }
}

__global__ void show_top(Tensor<float>* top) {
  size_t h = top->GetDims()[1];
  size_t w = top->GetDims()[2];
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      printf("%f ", top->at(0, i, j, 0));
    }
    printf("\n");
  }
}


void test_conv2d_cpu() {
  printf("Example code for conv2d cpu\n");
  size_t h = 400;
  size_t w = 400;

  Session* session = Session::GetNewSession();
  session->gpu = false;
 
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Conv2D<float> * conv_layer = new Conv2D<float>(15,15,1,1,2,new GaussianKernelInitializer<float>(15));
  const char* OUTPUT_BMP_PATH = "./tmp/test/out.bmp";

  size_t b_dims[4] = {1, h, w, 1};
  Tensor<float>* bottom = Tensor<float>::CreateTensorCPU(b_dims);
  size_t t_dims[4] = {1, h/2, w/2, 1};
  Tensor<float>* top = Tensor<float>::CreateTensorCPU(t_dims);

  for(int i = 0; i < h; i++) {
  	for(int j = 0; j < w; j++) {
  	  int b_idx[4] = {0, i, j, 0};
  	  bottom->at(b_idx) = (float) ((i+j) % 255);
  	}
  }
  conv_layer->Forward({bottom}, {top});

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

/*
void test_conv2d_gpu() {
  printf("Example code for conv2d gpu\n");
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  size_t h = 400;
  size_t w = 400;

  Session* session = Session::GetNewSession();
  session->gpu = true;

  size_t kernel = 15;
 
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Conv2D<float> * conv_layer = new Conv2D<float>(15,15,1,1,2,new GaussianKernelInitializer<float>(15), VALID);
  const char* OUTPUT_BMP_PATH = "./tmp/test/out_gpu.bmp";

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  size_t b_dims[4] = {1, h, w, 1};
  Tensor<float>* bottom = Tensor<float>::CreateTensorGPU(b_dims);
  
  size_t t_dims[4] = {1, h/2-kernel+1, w/2-kernel+1, 1};
  conv_layer->GetTopsDims({b_dims}, {t_dims});
  printf("%d %d %d %d \n", t_dims[0], t_dims[1], t_dims[2], t_dims[3]);
  Tensor<float>* top = Tensor<float>::CreateTensorGPU(t_dims);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  init_bottom<<<1,1>>>(bottom);
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  conv_layer->Forward({bottom}, {top});
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  show_top<<<1,1>>>(top);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
 
  Tensor<float> * top_cpu = Tensor<float>::TensorGPUtoCPU(top);
  cudaStatus = cudaGetLastError();
  
  checkCudaErrors(cudaStatus);
  
  bitmap_image img(w/2-kernel+1, h/2-kernel+1);	
  for (int i = 0; i < h/2-kernel+1; i++) {
    for (int j = 0; j < w/2-kernel+1; j++) {
      unsigned val = (unsigned) top_cpu->at(0, i, j, 0);
      img.set_pixel(j, i, val, val, val);
    }
  }
  img.save_image(OUTPUT_BMP_PATH);
  delete conv_layer;
  delete top_cpu;
  cudaFree(bottom);
  cudaFree(top);
}*/



void test_conv2d_gpu() {
  printf("Example code for conv2d gpu\n");
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  Session* session = Session::GetNewSession();
  session->gpu = true;
  session->batch_size = 64;

  size_t kernel = 15;
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Conv2D<float> conv_layer = Conv2D<float>(kernel,kernel,32,64,1, new GaussianKernelInitializer<float>(0.1), SAME);

  size_t b_dims[4] = {session->batch_size, 14, 14, 32};
  Tensor<float>* bottom = Tensor<float>::CreateTensorGPU(b_dims);
  init_bottom<<<1,1>>>(bottom);

  size_t t_dims[4];
  conv_layer.GetTopsDims({b_dims}, {t_dims});
  printf("%d %d %d %d \n", (int)b_dims[0], (int)b_dims[1], (int)b_dims[2], (int)b_dims[3]);
  printf("%d %d %d %d \n", (int)t_dims[0], (int)t_dims[1], (int)t_dims[2], (int)t_dims[3]);
  Tensor<float>* top = Tensor<float>::CreateTensorGPU(t_dims);
  checkCudaErrors(cudaGetLastError());

  startTimer();
  conv_layer.Forward({bottom}, {top});
  checkCudaErrors(cudaGetLastError());
  printf("conv layer forward: %3.4f ms \n", stopTimer()); 

  show_top<<<1,1>>>(top);
  cudaFree(top);
  cudaFree(bottom);
  checkCudaErrors(cudaGetLastError());
}

int main() {
  // test_conv2d_cpu();
  test_conv2d_gpu();
}
