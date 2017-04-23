
#include "layers/lrn.cu"
#include "basics/tensor.cu"
#include <vector>
#include <assert.h>

__global__ void initial_bottom(Tensor<float>* bottom) {
  const size_t* dims = bottom->GetDims();
  printf("(%d, %d)\n", int(dims[0]), int(dims[3]));
  for (int i = 0; i < int(dims[0]); ++i) {
    for (int j = 0; j < int(dims[3]); ++j) {
      bottom->at(i,0,0,j) = (float) i + j;
      printf("(%d, %d): %f\n", i, j, bottom->at(i,0,0,j));
    }
  }
}

__global__ void show_top(Tensor<float>* top) {
  printf("Printing top data\n");
  for (int i = 0; i < int(top->GetDims()[0]); ++i) {
    for (int j = 0; j < int(top->GetDims()[3]); ++j) {
      printf("(%d, %d): %f\n", i, j, top->at(i,0,0,j));
    }
  }
}

void test_lrn_cpu() {
  printf("Begin test lrn layer CPU\n");
  Session* session = Session::GetNewSession();
  session->gpu = false;
  session->batch_size = 2;


  size_t dims[4] = {2, 1, 1, 5};
  std::vector<Tensor<float>*> bottom;
  bottom.push_back(Tensor<float>::CreateTensorCPU(dims));
  std::vector<Tensor<float>*> top;
  top.push_back(Tensor<float>::CreateTensorCPU(dims));

  printf("(%d, %d)\n", bottom[0]->GetDims()[0], bottom[0]->GetDims()[3]);
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      bottom[0]->at(i,0,0,j) = (float) i + j;
      printf("(%d, %d): %f\n", i, j, bottom[0]->at(i,0,0,j));
    }
  }

  LRN<float> lrn_layer(3);
  lrn_layer.Forward(bottom, top);
  
  printf("Printing top data\n");
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      printf("(%d, %d): %f\n", i, j, top[0]->at(i,0,0,j));
    }
  }
}


void test_lrn_gpu() {
  printf("Begin test lrn layer GPU\n");
  Session* session = Session::GetNewSession();
  session->gpu = true;
  session->batch_size = 2;

  LRN<float> lrn_layer(3);

  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  size_t dims[4] = {2, 1, 1, 5};
  std::vector<Tensor<float>*> bottom;
  bottom.push_back(Tensor<float>::CreateTensorGPU(dims));
  std::vector<Tensor<float>*> top;
  top.push_back(Tensor<float>::CreateTensorGPU(dims));

  initial_bottom<<<1,1>>>(bottom[0]);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);

  lrn_layer.Forward(bottom, top);
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);

  show_top<<<1,1>>>(top[0]);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);
}


int main() {
  test_lrn_cpu();
  test_lrn_gpu();
}
