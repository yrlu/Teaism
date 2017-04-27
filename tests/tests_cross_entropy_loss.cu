
#include "layers/cross_entropy_loss.cu"
#include "basics/tensor.cu"
#include <vector>
#include <assert.h>

__global__ void initial_bottoms(Tensor<float>* bottoms_0, Tensor<float>* bottoms_1) {
  const size_t* dims =  bottoms_0->GetDims();
  printf("(%d, %d)\n", int(dims[0]), int(dims[3]));
  for (int i = 0; i < int(dims[0]); ++i) {
    for (int j = 0; j < int(dims[3]); ++j) {
      bottoms_0->at(i,0,0,j) = (float) (i + j + 1) / (int(dims[0]) + int(dims[3])+2);
      printf("(%d, %d): %f\n", i, j, bottoms_0->at(i,0,0,j));
    }
  }
  bottoms_1->at(0,0,0,0) = 1;
  bottoms_1->at(1,0,0,0) = 2;
}

__global__ void show_tops(Tensor<float>* tops) {
  printf("Printing tops data\n");
  printf("(%d, %d): %f\n", 0, 0, tops->at(0,0,0,0));
}

void test_cross_entropy_loss_cpu() {
  printf("Begin test cross-entropy loss layer CPU\n");

  int b = 2;
  int c = 3;

  Session* session = Session::GetNewSession();
  session->gpu = false;
  session->batch_size = b;


  size_t dims[4] = {b, 1, 1, c};
  std::vector<Tensor<float>*> bottoms;
  bottoms.push_back(Tensor<float>::CreateTensorCPU(dims));

  printf("(%d, %d)\n", bottoms[0]->GetDims()[0], bottoms[0]->GetDims()[3]);
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      bottoms[0]->at(i,0,0,j) = (float) (i + j + 1) / (dims[0] + dims[3]+2);
      printf("(%d, %d): %f\n", i, j, bottoms[0]->at(i,0,0,j));
    }
  }

  size_t dims_l[4] = {b, 1, 1, 1};
  bottoms.push_back(Tensor<float>::CreateTensorCPU(dims_l));
  bottoms[1]->at(0,0,0,0) = 1;
  bottoms[1]->at(1,0,0,0) = 2;

  size_t dims_t[4] = {1, 1, 1, 1};
  std::vector<Tensor<float>*> tops;
  tops.push_back(Tensor<float>::CreateTensorCPU(dims_t));

  CrossEntropyLoss<float> cross_entropy_loss_layer;
  cross_entropy_loss_layer.Forward(bottoms, tops);
  
  printf("Printing tops data\n");
  for (size_t i = 0; i < dims_t[0]; ++i) {
    for (size_t j = 0; j < dims_t[3]; ++j) {
      printf("(%d, %d): %f\n", i, j, tops[0]->at(i,0,0,j));
    }
  }


  std::vector<Tensor<float>*> bottoms_diff;
  bottoms_diff.push_back(Tensor<float>::CreateTensorCPU(dims));
  bottoms_diff.push_back(Tensor<float>::CreateTensorCPU(dims_l));

  cross_entropy_loss_layer.Backward(tops, tops, bottoms, bottoms_diff);

  printf("Printing bottoms diff\n");
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      printf("(%d, %d): %f\n", i, j, bottoms_diff[0]->at(i,0,0,j));
    }
  }


  delete bottoms[0], bottoms[1], tops[0], bottoms_diff[0], bottoms_diff[1];

}

void test_cross_entropy_loss_gpu() {
  printf("Begin test cross entropy loss layer GPU\n");
  Session* session = Session::GetNewSession();
  session->gpu = true;


  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  size_t dims[4] = {2, 1, 1, 3};
  std::vector<Tensor<float>*> bottoms;
  bottoms.push_back(Tensor<float>::CreateTensorGPU(dims));

  size_t dims_l[4] = {2, 1, 1, 1};
  bottoms.push_back(Tensor<float>::CreateTensorGPU(dims_l));


  initial_bottoms<<<1,1>>>(bottoms[0], bottoms[1]);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);

  size_t dims_t[4] = {1, 1, 1, 1};
  std::vector<Tensor<float>*> tops;
  tops.push_back(Tensor<float>::CreateTensorGPU(dims_t));

  CrossEntropyLoss<float> cross_entropy_loss_layer;
  cross_entropy_loss_layer.Forward(bottoms, tops);
 
  printf("Done GPU forward.\n");

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  show_tops<<<1,1>>>(tops[0]);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);
}


int main() {
  test_cross_entropy_loss_cpu();
  //test_cross_entropy_loss_gpu();
}
