#include <stdio.h>
#include "basics/tensor.cu"
#include <assert.h>
#include <cmath>
#include "basics/session.hpp"
#include "layers/relu.cu"


void test_relu_cpu() {
  printf("Example code for relu layer cpu\n");
  size_t h = 20;
  size_t w = 20;

  Session* session = Session::GetNewSession();
  session->gpu = false;
 
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Relu<float> * relu = new Relu<float>();

  const char* OUTPUT_BMP_PATH = "./tmp/test/out.bmp";
  size_t b_dims[4] = {1, h, w, 1};
  Tensor<float>* bottom = Tensor<float>::CreateTensorCPU(b_dims);
  size_t t_dims[4] = {1, h, w, 1};
  Tensor<float>* top = Tensor<float>::CreateTensorCPU(t_dims);

  for(int i = 0; i < h; i++) {
    for(int j = 0; j < w; j++) {
      int b_idx[4] = {0, i, j, 0};
      bottom->at(b_idx) = (float) (i-j);
    }
  }
  relu->Forward(bottom, top);

  for(int i = 0; i < h; i++) {
    for(int j = 0; j < w; j++) {
      printf("%d ", top->at(0, i, j, 0));
    }
    printf("\n")
  }
  delete relu;
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
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      printf("%f ", top->at(0, i, j, 0));
    }
    printf("\n");
  } 
  printf("%d \n", top->GetDataPtr());
}



void test_relu_gpu() {
  printf("Example code for relu layer gpu\n");
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  size_t h = 20;
  size_t w = 20;

  Session* session = Session::GetNewSession();
  session->gpu = true;
 
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Relu<float> * relu = new Relu<float>(2);

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
  
  relu->Forward(bottom, top);
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  show_top<<<1,1>>>(top);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
 
  Tensor<float> * top_cpu = Tensor<float>::TensorGPUtoCPU(top);
  cudaStatus = cudaGetLastError();
  
  checkCudaErrors(cudaStatus);

  delete relu;
  delete top_cpu;
  cudaFree(bottom);
  cudaFree(top);
}


int main() {
  test_relu_cpu();
  test_relu_gpu();
}



