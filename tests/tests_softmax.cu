
#include "layers/softmax.cu"
#include "basics/tensor.cu"
#include <vector>
#include <assert.h>

__global__ void initial_bottoms(Tensor<float>* bottoms) {
  const size_t* dims = bottoms->GetDims();
  printf("(%d, %d)\n", int(dims[0]), int(dims[3]));
  for (int i = 0; i < int(dims[0]); ++i) {
    for (int j = 0; j < int(dims[3]); ++j) {
      bottoms->at(i,0,0,j) = (float) i + j;
      printf("(%d, %d): %f\n", i, j, bottoms->at(i,0,0,j));
    }
  }
}

__global__ void show_tops(Tensor<float>* tops) {
  printf("Printing tops data\n");
  for (int i = 0; i < int(tops->GetDims()[0]); ++i) {
    for (int j = 0; j < int(tops->GetDims()[3]); ++j) {
      printf("(%d, %d): %f\n", i, j, tops->at(i,0,0,j));
    }
  }
}

__global__ void initial_top_diff(Tensor<float>* top_diff) {
  printf("Printing top diff\n");
  for (int i = 0; i < int(top_diff->GetDims()[0]); ++i) {
    for (int j = 0; j < int(top_diff->GetDims()[3]); ++j) {
      top_diff->at(i,0,0,j) = (float) i + j;
      printf("(%d, %d): %f\n", i, j, top_diff->at(i,0,0,j));
    }
  }
}

__global__ void show_bottom_diff(Tensor<float>* bottom_diff) {
  printf("Printing bottom diff\n");
  for (int i = 0; i < int(bottom_diff->GetDims()[0]); ++i) {
    for (int j = 0; j < int(bottom_diff->GetDims()[3]); ++j) {
      printf("(%d, %d): %f\n", i, j, bottom_diff->at(i,0,0,j));
    }
  }
}

void test_softmax_cpu() {
  printf("Begin test softmax layer CPU\n");
  Session* session = Session::GetNewSession();
  session->gpu = false;
  session->batch_size = 2;


  size_t dims[4] = {2, 1, 1, 3};
  std::vector<Tensor<float>*> bottoms;
  bottoms.push_back(Tensor<float>::CreateTensorCPU(dims));
  std::vector<Tensor<float>*> tops;
  tops.push_back(Tensor<float>::CreateTensorCPU(dims));

  printf("(%d, %d)\n", bottoms[0]->GetDims()[0], bottoms[0]->GetDims()[3]);
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      bottoms[0]->at(i,0,0,j) = (float) i + j;
      printf("(%d, %d): %f\n", i, j, bottoms[0]->at(i,0,0,j));
    }
  }

  Softmax<float> softmax_layer;
  softmax_layer.Forward(bottoms, tops);
  
  printf("Printing tops data\n");
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      printf("(%d, %d): %f\n", i, j, tops[0]->at(i,0,0,j));
    }
  }

  std::vector<Tensor<float>*> tops_diff;
  tops_diff.push_back(Tensor<float>::CreateTensorCPU(dims));
  printf("Printing tops diff\n");
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      tops_diff[0]->at(i,0,0,j) = (float) i + j;
      printf("(%d, %d): %f\n", i, j, tops_diff[0]->at(i,0,0,j));
    }
  }

  std::vector<Tensor<float>*> bottoms_diff;
  bottoms_diff.push_back(Tensor<float>::CreateTensorCPU(dims));

  softmax_layer.Backward(tops, tops_diff, bottoms, bottoms_diff);

  printf("Printing bottoms_diff data\n");
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[3]; ++j) {
      printf("(%d, %d): %f\n", i, j, bottoms_diff[0]->at(i,0,0,j));
    }
  }


  delete bottoms[0], bottoms_diff[0], tops[0], tops_diff[0];

}


void test_softmax_gpu() {
  printf("Begin test softmax layer GPU\n");
  Session* session = Session::GetNewSession();
  session->gpu = true;
  session->batch_size = 2;

  Softmax<float> softmax_layer;

  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  size_t dims[4] = {2, 1, 1, 3};
  std::vector<Tensor<float>*> bottoms;
  bottoms.push_back(Tensor<float>::CreateTensorGPU(dims));
  std::vector<Tensor<float>*> tops;
  tops.push_back(Tensor<float>::CreateTensorGPU(dims));

  initial_bottoms<<<1,1>>>(bottoms[0]);

  softmax_layer.Forward(bottoms, tops);
  
  show_tops<<<1,1>>>(tops[0]);

  std::vector<Tensor<float>*> tops_diff;
  tops_diff.push_back(Tensor<float>::CreateTensorGPU(dims));

  initial_top_diff<<<1,1>>>(tops_diff[0]);

  std::vector<Tensor<float>*> bottoms_diff;
  bottoms_diff.push_back(Tensor<float>::CreateTensorGPU(dims));

  softmax_layer.Backward(tops, tops_diff, bottoms, bottoms_diff);

  show_bottom_diff<<<1,1>>>(bottoms_diff[0]);

  cudaFree(bottoms[0]);
  cudaFree(bottoms_diff[0]);
  cudaFree(tops[0]);
  cudaFree(tops_diff[0]);

}


int main() {
  test_softmax_cpu();
  test_softmax_gpu();
}
