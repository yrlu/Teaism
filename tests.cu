#include <stdio.h>
#include "basics/tensor.cu"
#include "initializers/gaussian_kernel_initializer.cu"
#include <assert.h>
#include <cmath>


__global__ allocate_tensor_dataarray(Tensor<float> * tensor_gpu) {
  tensor_gpu->AllocateDataArray();
  for(int i = 0; i < tensor_gpu->size(); i++) {
    tensor_gpu->GetDataPtr()[i] = i;
  }
}

__global__ access_tensor_dataarray(Tensor<float> * tensor_gpu) {
  for(int i = 0; i < tensor_gpu->size(); i++) {
    printf("%f \n", tensor_gpu->GetDataPtr()[i]);
  }
}

void test_tensor() {
  size_t dims[4] = {3, 3, 3, 3};
  Tensor<float>* tensor_cpu = new Tensor<float>(dims);
  Tensor<float>* tensor_gpu;
  cudaMalloc((void **)&tensor_gpu, sizeof(Tensor<float>));
  cudaMemcpy(tensor_gpu, tensor_cpu, sizeof(Tensor<float>), cudaMemcpyHostToDevice);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  allocate_tensor_memory<<<1,1>>>(tensor_gpu);
  access_tensor_dataarray<<<1, 1>>>(tensor_gpu);
}

int main(void) {
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);
  test_tensor();  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);
}