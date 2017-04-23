#include <stdio.h>
#include "basics/tensor.cu"
#include "initializers/const_initializer.cu"
#include <assert.h>
#include <cmath>

void test_const_init_cpu() {
  size_t w_dims[4] = {5,5,1,1};
  Tensor<float> * W_cpu = Tensor<float>::CreateTensorCPU(w_dims);
  size_t b_dims[4] = {1, 1, 1, 1};
  Tensor<float> * b_cpu = Tensor<float>::CreateTensorCPU(b_dims);
  ConstInitializer<float>(5.0, 1.0).Initialize(W_cpu, b_cpu, false);

  for(int i = 0; i < 5; i++) {
    for(int j = 0; j < 5; j++) {
      int idx[4] = {i, j, 0, 0};
      printf("%f ", W_cpu->at(idx));
    }
    printf("\n");
  }
  delete W_cpu;
  delete b_cpu;
}

__global__ void print_kernel(Tensor<float> * W, Tensor<float> *b) {
  for(int i = 0; i < 5; i++) {
    for(int j = 0; j < 5; j++) {
      int idx[4] = {i, j, 0, 0};
      printf("%f ", W->at(idx));
    }
    printf("\n");
  }
  printf("%f \n", b->GetDataPtr()[0]);
}


void test_const_init_gpu() {
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  size_t w_dims[4] = {5,5,1,1};
  Tensor<float> * W_gpu = Tensor<float>::CreateTensorGPU(w_dims);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  size_t b_dims[4] = {1, 1, 1, 1};
  Tensor<float> * b_gpu = Tensor<float>::CreateTensorGPU(b_dims);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  ConstInitializer<float>(5.0, 1.0).Initialize(W_gpu, b_gpu, true);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  print_kernel<<<1,1>>>(W_gpu, b_gpu);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);
}

int main(void) {
  test_const_init_gpu();  
  test_const_init_cpu();
}