#ifndef TENSOR_CUH_
#define TENSOR_CUH_

#include <assert.h>
#include <cstdlib>
#include <numeric>
#include <functional>
#include "basics/session.hpp"
#include "cuda_runtime.h"
#include "utils/helper_cuda.h"
#include "stdio.h"

/* 
4D Tensor
*/
template<class Dtype>
class Tensor {

public:

  __host__ __device__ unsigned GetIdx(const int* idx) const {
    unsigned out_idx = 0;
    for (int i = 0; i < 4; i++)
      out_idx = out_idx*dims_[i] + idx[i];
    return out_idx;
  }

  __host__ __device__ const size_t* GetDims() const {
    return dims_;
  }

  __host__ __device__ Dtype* GetDataPtr() const {
    return data_array_;
  }

  __host__ __device__ void SetDataPtr(Dtype* data_array_ptr) {
    data_array_ = data_array_ptr;
  }

  __host__ __device__ Dtype& at(const int i0, const int i1, const int i2, const int i3) {
    int idx[4] = {i0, i1, i2, i3};
    return at(idx);
  }

  __host__ __device__ Dtype& at(const int* idx) {
    assert(isValidIdx(idx));
    return data_array_[GetIdx(idx)];
  }

  __host__ __device__ const Dtype atPadding(const int i0, const int i1, const int i2, const int i3) {
    int idx[4] = {i0, i1, i2, i3};
    return atPadding(idx);
  }

  __host__ __device__ const Dtype atPadding(int* idx, Dtype default_val = 0.0) const {
    if (!isValidIdx(idx)) return default_val;
    return data_array_[GetIdx(idx)];
  }

  __host__ __device__ bool isValidIdx(const int i0, const int i1, const int i2, const int i3) {
    int idx[4] = {i0, i1, i2, i3};
    return isValidIdx(idx);
  }

  __host__ __device__ bool isValidIdx(const int* idx) const {
    for(int i = 0; i < 4; i++) {
      if(idx[i] < 0 || idx[i] >= dims_[i]) return false;
    }
    return true;
  }

  __host__ __device__ size_t size() const {
    return len_;
  }

  __host__ static Tensor<Dtype>* CreateTensorGPU(size_t* dims, bool allocate_memory = true);
  __host__ static Tensor<Dtype>* CreateTensorCPU(size_t* dims, bool allocate_memory = true);
  __host__ static Tensor<Dtype> * TensorGPUtoCPU(Tensor<Dtype> * tensor_gpu);
  __host__ static Tensor<Dtype> * TensorCPUtoGPU(Tensor<Dtype> * tensor_cpu);
  __host__ static void AllocateDataArrayGPU(Tensor<Dtype> * tensor_gpu);
  __host__ static void AllocateDataArrayCPU(Tensor<Dtype> * tensor_cpu);

  __host__ __device__ ~Tensor() {
    if(data_array_ != NULL) {
      delete [] data_array_;
    }
  }
  

private:
  __host__ __device__ Tensor(size_t dims[4]): data_array_(NULL) {
    len_ = dims[0] * dims[1] * dims[2] * dims[3];
    dims_[0] = dims[0];
    dims_[1] = dims[1];
    dims_[2] = dims[2];
    dims_[3] = dims[3];
  }

  Dtype* data_array_;
  size_t dims_[4];
  size_t len_;
};


// Create CPU/GPU Tensor
template<class Dtype>
__host__ Tensor<Dtype>* Tensor<Dtype>::CreateTensorCPU(size_t* dims, bool allocate_memory) {
  Tensor<Dtype> * tensor_cpu = new Tensor(dims);
  if (allocate_memory) {
    AllocateDataArrayCPU(tensor_cpu);
  }
  return tensor_cpu;
}

template<class Dtype>
__host__ Tensor<Dtype>* Tensor<Dtype>::CreateTensorGPU(size_t* dims, bool allocate_memory) {
  Tensor<Dtype> tensor_cpu(dims);
  Tensor<Dtype>* tensor_gpu;
  cudaMalloc((void**)&tensor_gpu, sizeof(Tensor<Dtype>));
  cudaMemcpy(tensor_gpu, &tensor_cpu, sizeof(Tensor<Dtype>), cudaMemcpyHostToDevice);

  if (allocate_memory) {
    AllocateDataArrayGPU(tensor_gpu);
  }
  return tensor_gpu;
}

template<class Dtype>
__host__ Tensor<Dtype> * Tensor<Dtype>::TensorGPUtoCPU(Tensor<Dtype> * tensor_gpu) {
  Tensor<Dtype> * tensor_cpu = (Tensor<Dtype> *)malloc(sizeof(Tensor<Dtype>));
  cudaMemcpy(tensor_cpu, tensor_gpu, sizeof(Tensor<Dtype>), cudaMemcpyDeviceToHost);
  Dtype * data_array_ = (Dtype*) malloc(tensor_cpu->size()*sizeof(Dtype));
  cudaMemcpy(data_array_, tensor_cpu->GetDataPtr(), tensor_cpu->size() * sizeof(Dtype), cudaMemcpyDeviceToHost);
  tensor_cpu->SetDataPtr(data_array_);
  return tensor_cpu;
}

template<class Dtype>
__host__ Tensor<Dtype> * Tensor<Dtype>::TensorCPUtoGPU(Tensor<Dtype> * tensor_cpu) {
  Tensor<Dtype> * tensor_gpu; 
  cudaMalloc((void **)&tensor_gpu, sizeof(Tensor<Dtype>));
  cudaMemcpy(tensor_gpu, tensor_cpu, sizeof(Tensor<Dtype>), cudaMemcpyHostToDevice);

  Dtype* data_array;
  cudaMalloc((void**) &data_array, sizeof(Dtype)*tensor_cpu->size());
  cudaMemcpy(data_array, tensor_cpu->GetDataPtr(), sizeof(Dtype)*tensor_cpu->size(), cudaMemcpyHostToDevice);
  cudaMemcpy(&tensor_gpu->data_array_, &data_array, sizeof(Dtype*), cudaMemcpyHostToDevice);

  return tensor_gpu;
}

// Allocate Memory 
template<class Dtype>
__host__ void Tensor<Dtype>::AllocateDataArrayGPU(Tensor<Dtype> * tensor_gpu) {
    size_t * len = (size_t *) malloc(sizeof(size_t));
    cudaMemcpy(len, &tensor_gpu->len_, sizeof(size_t), cudaMemcpyDeviceToHost);
    Dtype* data_array_gpu;
    cudaMalloc((void**)&(data_array_gpu), sizeof(Dtype)*(*len));
    cudaMemcpy(&(tensor_gpu->data_array_), &data_array_gpu, sizeof(Dtype*), cudaMemcpyHostToDevice);
}

template<class Dtype>
__host__ void Tensor<Dtype>::AllocateDataArrayCPU(Tensor<Dtype> * tensor_cpu) {
  if (tensor_cpu->data_array_ == NULL) {
    tensor_cpu->data_array_ = new Dtype[tensor_cpu->len_];
  }
}


#endif // TENSOR_CUH_
