#include <stdio.h>
#include "basics/tensor.cu"
#include "initializers/gaussian_kernel_initializer.cu"
#include <assert.h>
#include <cmath>


cudaError_t cudaStatus = cudaSetDevice(0);

__global__ void access_tensor_dataarray(Tensor<float> * tensor_gpu) {
  const size_t* dims = tensor_gpu->GetDims();
  printf("%d %d %d %d \n", (int)tensor_gpu->GetDims()[0], (int)tensor_gpu->GetDims()[1], (int)tensor_gpu->GetDims()[2], (int)tensor_gpu->GetDims()[3]);
  
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

  printf("%d \n", tensor_gpu->size()*sizeof(float));
  printf("%d \n", tensor_gpu->GetDataPtr());
}

void test_tensor_gpu() {
  printf("-- Example usage of tensor on gpu --\n");
  size_t dims[4] = {1, 3, 3, 1};

  Tensor<float>* tensor_gpu = Tensor<float>::CreateTensorGPU(dims);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  // Tensor<float>::AllocateDataArrayGPU(tensor_gpu);
  access_tensor_dataarray<<<1, 1>>>(tensor_gpu);
  
  cudaFree(tensor_gpu);
}


void test_tensor_gpu_to_cpu() {
  size_t dims[4] = {1,3,3,1};
  Tensor<float>* tensor_gpu = Tensor<float>::CreateTensorGPU(dims);
//  Tensor<float>::AllocateDataArrayGPU(tensor_gpu);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  access_tensor_dataarray<<<1, 1>>>(tensor_gpu);
  
  Tensor<float> * tensor_cpu = (Tensor<float> *)malloc(sizeof(Tensor<float>));
  cudaMemcpy(tensor_cpu, tensor_gpu, sizeof(Tensor<float>), cudaMemcpyDeviceToHost);
  printf("%d \n", tensor_cpu->size() * sizeof(float));
  printf("%d \n", tensor_cpu->GetDataPtr());

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  float * data_array_cpu;
  data_array_cpu = (float*) malloc(tensor_cpu->size()*sizeof(float));
  cudaMemcpy(data_array_cpu, tensor_cpu->GetDataPtr(), tensor_cpu->size()*sizeof(float), cudaMemcpyDeviceToHost);


  Tensor<float> * tensor_cpu2 = Tensor<float>::TensorGPUtoCPU(tensor_gpu);
  for(int i0 = 0; i0 < dims[0]; i0++) {
    for(int i1 = 0; i1 < dims[1]; i1++) {
      for(int i2 = 0; i2 <dims[2]; i2++) {
        for(int i3 = 0; i3 < dims[3]; i3++) {
          int idx[4] = {i0, i1, i2, i3};
          printf("%f \n", tensor_cpu2->at(idx));
          tensor_cpu2->at(idx) = 0;
        }
      }
    }
  }
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
}

void test_tensor_cpu() {
  printf("-- Example usage of tensor on cpu --\n");
  size_t dims[4] = {1,3,3,1};
  Tensor<float> * tensor_cpu = Tensor<float>::CreateTensorCPU(dims);
//  Tensor<float>::AllocateDataArrayCPU(tensor_cpu);

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

  for(int i = 0; i < tensor_cpu->size(); i++) {
    printf("%f \n", tensor_cpu->GetDataPtr()[i]);
  }
  delete tensor_cpu;
}

void test_reshape_cpu() {
  size_t dims[4] = {1,3,3,1};
  Tensor<float> * tensor_cpu = Tensor<float>::CreateTensorCPU(dims);
  dims[1] = 9;
  dims[2] = 1;
  Tensor<float>::ReshapeTensorCPU(tensor_cpu, dims);


  printf("%d %d %d %d\n", tensor_cpu->GetDims()[0], tensor_cpu->GetDims()[1], tensor_cpu->GetDims()[2], tensor_cpu->GetDims()[3]);
  delete tensor_cpu;
}


__global__ void show_dims(Tensor<float> *tensor_gpu) {
  printf("%d %d %d %d\n", (int)tensor_gpu->GetDims()[0], (int)tensor_gpu->GetDims()[1], (int)tensor_gpu->GetDims()[2], (int)tensor_gpu->GetDims()[3]); 
}

void test_reshape_gpu() {
  size_t dims[4] = {1,3,3,1};
  Tensor<float> * tensor_gpu = Tensor<float>::CreateTensorGPU(dims);
  dims[1] = 9;
  dims[2] = 1;
  Tensor<float>::ReshapeTensorGPU(tensor_gpu, dims);

  checkCudaErrors(cudaGetLastError());

  show_dims<<<1, 1>>>(tensor_gpu);
  cudaFree(tensor_gpu);
  
  checkCudaErrors(cudaGetLastError());
}


void test_indices() {
  size_t dims[4] = {3,3,3,3};
  Tensor<float> * tensor_cpu = Tensor<float>::CreateTensorCPU(dims);
  for(int i = 0; i < tensor_cpu->size(); i++) {
    tensor_cpu->GetDataPtr()[i] = i;
  }

  for(int i = 0; i < 3; i++) {
    printf("%f \n", tensor_cpu->at(0,0,0,i));
  }

  delete tensor_cpu;
}

void test_dims() {
  size_t dims[4] = {3,3,3,3};
  Tensor<float> * tensor_gpu = Tensor<float>::CreateTensorGPU(dims);
  size_t dims2[4];
  Tensor<float>::GetTensorGPUDims(tensor_gpu, dims2);
  printf("%d %d %d %d \n", dims2[0], dims2[1], dims2[2], dims2[3]);
  checkCudaErrors(cudaGetLastError());
}


int main(void) {
  // tensor
  checkCudaErrors(cudaStatus);
  test_tensor_gpu();
  // test_tensor_gpu_to_cpu();
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);
//  test_tensor_cpu();  
  test_reshape_gpu();
  test_reshape_cpu();
  test_indices();
  test_dims();
}
