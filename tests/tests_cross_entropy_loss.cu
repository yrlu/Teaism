
#include "layers/cross_entropy_loss.cu"
#include "basics/tensor.cu"
#include <vector>
#include <assert.h>

__global__ void initial_bottom(Tensor<float>* bottom_0, Tensor<float>* bottom_1) {
  const size_t* dims =  bottom_0->GetDims();
  printf("(%d, %d)\n", int(dims[0]), int(dims[3]));
  for (int i = 0; i < int(dims[0]); ++i) {
    for (int j = 0; j < int(dims[3]); ++j) {
      bottom_0->at(i,0,0,j) = (float) (i + j + 1) / (int(dims[0]) + int(dims[3])+2);
      printf("(%d, %d): %f\n", i, j, bottom_0->at(i,0,0,j));
    }
  }
  bottom_1->at(0,0,0,0) = 1;
  bottom_1->at(1,0,0,0) = 2;
}

__global__ void show_top(Tensor<float>* top) {
  printf("Printing top data\n");
  printf("(%d, %d): %f\n", 0, 0, top->at(0,0,0,0));
}

void test_cross_entropy_loss_cpu() {
  printf("Begin test cross-entropy loss layer CPU\n");
  Session* session = Session::GetNewSession();
  session->gpu = false;

  size_t dims[4] = {2, 1, 1, 3};
  std::vector<Tensor<float>*> bottom;
  bottom.push_back(Tensor<float>::CreateTensorCPU(dims));

  printf("(%d, %d)\n", bottom[0]->GetDims()[0], bottom[0]->GetDims()[3]);
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      bottom[0]->at(i,0,0,j) = (float) (i + j + 1) / (dims[0] + dims[3]+2);
      printf("(%d, %d): %f\n", i, j, bottom[0]->at(i,0,0,j));
    }
  }

  size_t dims_l[4] = {2, 1, 1, 1};
  bottom.push_back(Tensor<float>::CreateTensorCPU(dims_l));
  bottom[1]->at(0,0,0,0) = 1;
  bottom[1]->at(1,0,0,0) = 2;

  size_t dims_t[4] = {1, 1, 1, 1};
  std::vector<Tensor<float>*> top;
  top.push_back(Tensor<float>::CreateTensorCPU(dims_t));

  CrossEntropyLoss<float> cross_entropy_loss_layer;
  cross_entropy_loss_layer.Forward(bottom, top);
  
  printf("Printing top data\n");
  for (size_t i = 0; i < dims_t[0]; ++i) {
    for (size_t j = 0; j < dims_t[3]; ++j) {
      printf("(%d, %d): %f\n", i, j, top[0]->at(i,0,0,j));
    }
  }
}

void test_cross_entropy_loss_gpu() {
  printf("Begin test cross entropy loss layer GPU\n");
  Session* session = Session::GetNewSession();
  session->gpu = true;


  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  size_t dims[4] = {2, 1, 1, 3};
  std::vector<Tensor<float>*> bottom;
  bottom.push_back(Tensor<float>::CreateTensorGPU(dims));

  size_t dims_l[4] = {2, 1, 1, 1};
  bottom.push_back(Tensor<float>::CreateTensorGPU(dims_l));


  initial_bottom<<<1,1>>>(bottom[0], bottom[1]);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);

  size_t dims_t[4] = {1, 1, 1, 1};
  std::vector<Tensor<float>*> top;
  top.push_back(Tensor<float>::CreateTensorGPU(dims_t));

  CrossEntropyLoss<float> cross_entropy_loss_layer;
  cross_entropy_loss_layer.Forward(bottom, top);
 
  printf("Done GPU forward.\n");

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  show_top<<<1,1>>>(top[0]);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);
}


int main() {
  test_cross_entropy_loss_cpu();
  test_cross_entropy_loss_gpu();
}
