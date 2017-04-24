#include <stdio.h>
#include "basics/tensor.cu"
#include <assert.h>
#include <vector>
#include "basics/session.hpp"
#include "initializers/const_initializer.cu"
#include "layers/fc.cu"
#include "utils/utils.cu"

__global__ void init_bottom(Tensor<float> * bottom) {
  for(int b = 0; b < bottom->GetDims()[0]; b++) {
    for(int i = 0; i < bottom->GetDims()[3]; i++) {
      bottom->at(b, 0, 0, i) = i+b;
    }
  }
}

__global__ void show_top(Tensor<float>* top) {
  for(int b = 0; b < top->GetDims()[0]; b++) {
    for(int i = 0; i < top->GetDims()[3]; i++) {
      printf("%f ", top->at(b, 0, 0, i));
    }
    printf("\n");
  }
}

void test_fc_gpu() {
  printf("Example code for fully connected layer on gpu\n");
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  size_t in_channels = 64;
  size_t out_channels = 10;

  Session* session = Session::GetNewSession();
  session->gpu = true;
  session->batch_size = 64;

  ConstInitializer<float> const_init(2.0, 1.0);
  FC<float> fc(in_channels, out_channels, &const_init);

  size_t b_dims[4] = {session->batch_size, 1, 1, in_channels};
  Tensor<float>* bottom = Tensor<float>::CreateTensorGPU(b_dims);
  size_t t_dims[4] = {session->batch_size, 1, 1, out_channels};
  Tensor<float>* top = Tensor<float>::CreateTensorGPU(t_dims);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  init_bottom<<<1, 1>>>(bottom);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  startTimer();  
  fc.Forward({bottom}, {top});
  printf("fc layer forward: %3.4f ms \n", stopTimer()); 

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  show_top<<<1,1>>>(top);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  cudaFree(bottom);
  cudaFree(top);
}



void test_fc_cpu() {
  printf("Example code for fully connected layer on cpu\n");

  size_t in_channels = 64;
  size_t out_channels = 10;

  Session* session = Session::GetNewSession();
  session->gpu = false;
  session->batch_size = 64;

  ConstInitializer<float> const_init(2.0, 1.0);
  FC<float> fc(in_channels, out_channels, &const_init);

  size_t b_dims[4] = {session->batch_size, 1, 1, in_channels};
  Tensor<float>* bottom = Tensor<float>::CreateTensorCPU(b_dims);
  size_t t_dims[4] = {session->batch_size, 1, 1, out_channels};
  Tensor<float>* top = Tensor<float>::CreateTensorCPU(t_dims);

  for(int b = 0; b < session->batch_size; b++) {
    for(int i = 0; i < in_channels; i++) {
      bottom->at(b, 0, 0, i) = b+i;
    }
  }

  startTimer();
  fc.Forward({bottom}, {top});
  printf("conv layer forward: %3.4f ms \n", stopTimer()); 

  for(int b = 0; b < session->batch_size; b++) {
    for(int i = 0; i < out_channels; i++) {
      printf("%f ", top->at(b, 0, 0, i));
    }
    printf("\n");
  }

  delete bottom;
  delete top;
}



int main() {
  test_fc_cpu();
  test_fc_gpu();
}
