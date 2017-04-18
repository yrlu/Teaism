#include <stdio.h>
#include "basics/tensor.cu"
#include "initializers/gaussian_kernel_initializer.cu"
#include <assert.h>
#include <cmath>


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

//  float * data_array_gpu = NULL;
//  cudaMalloc(&data_array_gpu, tensor_cpu->size()*sizeof(float));


//  Tensor<float> * tensor_cpu = Tensor<float>::TensorGPUtoCPU(tensor_gpu);
/*    Tensor<Dtype> * tensor_cpu = (Tensor<Dtype> *)malloc(sizeof(Tensor<Dtype>));
    cudaMemcpy(tensor_cpu, tensor_gpu, sizeof(Tensor<Dtype>), cudaMemcpyDeviceToHost);
    // TODO:
    Dtype * data_array_ = (Dtype*) malloc(tensor_cpu->size()*sizeof(Dtype));
    cudaMemcpy(data_array_, tensor_gpu->GetDataPtr(), tensor_cpu->size() * sizeof(Dtype), cudaMemcpyDeviceToHost);
    tensor_cpu->SetDataPtr(data_array_);
    return tensor_cpu;
*/
  // TODO
  // cudaMemcpy(data_array_gpu, data_array_gpu,  tensor_cpu->size()*sizeof(float), cudaMemcpyDeviceToDevice);
  // float* data_array_ptr_gpu;
  // cudaMemcpy(&data_array_ptr_gpu, &tensor_gpu->data_array_, sizeof(float*), cudaMemcpyDeviceToHost);
  // cudaMemcpy(data_array_gpu, data_array_ptr_gpu, tensor_cpu->size()*sizeof(float), cudaMemcpyDeviceToDevice);
  // printf("%d \n", data_array_ptr_gpu);
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
}

void test_tensor_cpu() {
  printf("-- Example usage of tensor on cpu --\n");
  size_t dims[4] = {1,3,3,1};
  Tensor<float> * tensor_cpu = Tensor<float>::CreateTensorCPU(dims);
  // tensor_cpu->AllocateDataArray();

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


int main(void) {
  // tensor
  checkCudaErrors(cudaStatus);
//  test_tensor_gpu();
  test_tensor_gpu_to_cpu();
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);
//  test_tensor_cpu();
}
