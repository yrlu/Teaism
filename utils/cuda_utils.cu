#ifndef CUDA_UTILS_CU
#define CUDA_UTILS_CU



__global__ void allocate_tensor_dataarray(Tensor<float> * tensor_gpu) {
  tensor_gpu->AllocateDataArray();
}



#endif // CUDA_UTILS_CU