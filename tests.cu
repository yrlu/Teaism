#include <stdio.h>
#include "basics/tensor.cu"
#include "initializers/gaussian_kernel_initializer.cu"
#include <assert.h>
#include <cmath>
#include "utils/cuda_utils.cu"



cudaError_t cudaStatus = cudaSetDevice(0);

__global__ void access_tensor_dataarray(Tensor<float> * tensor_gpu) {
  const size_t* dims = tensor_gpu->GetDims();
  for(int i = 0; i < 4; i++) {
    printf("%d \n", dims[i]);
  }

  for(int i = 0; i < tensor_gpu->size(); i++) {
    tensor_gpu->GetDataPtr()[i] = i;
  }

  for(int i0 = 0; i0 < dims[0]; i0++) {
    for(int i1 = 0; i1 < dims[1]; i1++) {
      for(int i2 = 0; i2 <dims[2]; i2++) {
        for(int i3 = 0; i3 < dims[3]; i3++) {
          int idx[4] = {i0, i1, i2, i3};
          printf("%f \n", tensor_gpu->at(idx));
          tensor_gpu->at(idx) = 0;
        }
      }
    }
  }

  for(int i = 0; i < tensor_gpu->size(); i++) {
    printf("%f \n", tensor_gpu->GetDataPtr()[i]);
  }
}

void test_tensor_gpu() {
  size_t dims[4] = {3, 3, 3, 3};
  Tensor<float>* tensor_cpu = new Tensor<float>(dims);
  Tensor<float>* tensor_gpu;
  cudaMalloc((void **)&tensor_gpu, sizeof(Tensor<float>));
  cudaMemcpy(tensor_gpu, tensor_cpu, sizeof(Tensor<float>), cudaMemcpyHostToDevice);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  allocate_tensor_dataarray<<<1, 1>>>(tensor_gpu);
  access_tensor_dataarray<<<1, 1>>>(tensor_gpu);
  delete tensor_cpu;
  cudaFree(tensor_gpu);
}


void test_tensor_cpu() {
  size_t dims[4] = {3,3,3,3};
  Tensor<float> * tensor_cpu = new Tensor<float>(dims);
  tensor_cpu->AllocateDataArray();

  for(int i = 0; i < tensor_cpu->size(); i++) {
    tensor_cpu->GetDataPtr()[i] = i;
  }

  for(int i0 = 0; i0 < dims[0]; i0++) {
    for(int i1 = 0; i1 < dims[1]; i1++) {
      for(int i2 = 0; i2 <dims[2]; i2++) {
        for(int i3 = 0; i3 < dims[3]; i3++) {
          int idx[4] = {i0, i1, i2, i3};
          printf("%f \n", tensor_cpu->at(idx));
          tensor_cpu->at(idx) = 0;
        }
      }
    }
  }
  delete tensor_cpu;
}


int main(void) {  
  checkCudaErrors(cudaStatus);
  test_tensor_gpu();  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);


  test_tensor_cpu();
}