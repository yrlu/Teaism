
#include "layers/softmax.cu"
#include "basics/tensor.cu"
#include <vector>
#include <assert.h>

/*
__global__ void show_top(Tensor<float>* top) {
  printf("%f\n", top->at(0,1020,531,0));
  printf("%f\n", top->at(1,1020,531,0));
  printf("%f\n", top->at(0,1001,555,1));
  printf("%f\n", top->at(1,1001,555,1));
  printf("%f\n", top->at(0,1000,500,2));
  printf("%f\n", top->at(1,1000,500,2));
}

__global__ void show_top_label(Tensor<float>* top) {
  printf("%f\n", top->at(0,0,0,0));
  printf("%f\n", top->at(1,0,0,0));
}
*/

void test_softmax_cpu() {
  printf("Begin test softmax layer CPU\n");
  Session* session = Session::GetNewSession();
  session->gpu = false;

  Softmax<float> softmax_layer();

  size_t dims[4] = {2, 1, 1, 3};
  std::vector<Tensor<float>*> bottom;
  bottom.push_back(Tensor<float>::CreateTensorCPU(dims));
  std::vector<Tensor<float>*> top;
  top.push_back(Tensor<float>::CreateTensorCPU(dims));

  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      bottom[0]->at(i,0,0,j) = i+j;
      printf("(%d, %d): %d\n", i, j, bottom[0]->at(i,0,0,j));
    }
  }

//  softmax_layer.Forward(bottom, top);
  
  printf("Printing bottom data\n");
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      printf("(%d, %d): %d\n", i, j, bottom[0]->at(i,0,0,j));
    }
  }
  printf("Printing top data\n");
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      printf("(%d, %d): %d\n", i, j, top[0]->at(i,0,0,j));
    }
  }
}

/*
void test_softmax_gpu() {
  printf("Begin test softmax layer GPU\n");
  Session* session = Session::GetNewSession();
  session->gpu = true;

  Softmax<float> softmax_layer();


  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  std::vector<Tensor<float>* > top;
  top = softmax_layer.Forward();
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  show_top_label<<<1,1>>>(top[1]);
  show_top<<<1,1>>>(top[0]);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);
}
*/

int main() {
  test_softmax_cpu();
//  test_softmax_gpu();
}
