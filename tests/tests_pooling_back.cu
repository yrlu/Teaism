#include <stdio.h>
#include "basics/tensor.cu"
#include <assert.h>
#include <cmath>
#include "basics/session.hpp"
#include "layers/pooling.cu"
#include "utils/bitmap_image.hpp"


void test_pooling_cpu() {
  printf("Start testing pooling layer CPU\n");
  size_t b = 2;
  size_t h = 4;
  size_t w = 4;
  size_t c = 2;

  Session* session = Session::GetNewSession();
  session->gpu = false;
  session->batch_size = b;
 
  Pooling<float> pooling_layer(2, MAX, 2);

  size_t dims_b[4] = {b, h, w, c};
  Tensor<float>* bottom = Tensor<float>::CreateTensorCPU(dims_b);

  printf("Initialize bottom tensor...\n");
  for (int i = 0; i < b; ++i) {
    for (int l = 0; l < c; ++l) {
      for (int j = 0; j < h; ++j) {
        for (int k = 0; k < w; ++k) {
          int idx[4] = {i, j, k, l};
          bottom->at(idx) = (float)(i+j+k+l)/255;
          printf("%f ", bottom->at(idx));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }

  size_t dims_t[4] = {b,0,0,c};
  pooling_layer.GetTopsDims({dims_b}, {dims_t});
  Tensor<float>* top = Tensor<float>::CreateTensorCPU(dims_t);

  pooling_layer.Forward({bottom}, {top});
  printf("Done forward.\n");

  printf("Show top tensor...\n");
  for (int i = 0; i < b; ++i) {
    for (int l = 0; l < c; ++l) {
      for (int j = 0; j < dims_t[1]; ++j) {
        for (int k = 0; k < dims_t[2]; ++k) {
          int idx[4] = {i, j, k, l};
          printf("%f ", top->at(idx));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }

  Tensor<float>* top_diff = Tensor<float>::CreateTensorCPU(dims_t);
  printf("Initialize top diff...\n");
  for (int i = 0; i < b; ++i) {
    for (int l = 0; l < c; ++l) {
      for (int j = 0; j < dims_t[1]; ++j) {
        for (int k = 0; k < dims_t[2]; ++k) {
          int idx[4] = {i, j, k, l};
          top_diff->at(idx) = float (i+j+k+l);
          printf("%f ", top_diff->at(idx));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }

  Tensor<float>* bottom_diff = Tensor<float>::CreateTensorCPU(dims_b);

  pooling_layer.Backward({top},{top_diff},{bottom},{bottom_diff});
  printf("Done backward.\n");

  printf("Show bottom diff...\n");
  for (int i = 0; i < b; ++i) {
    for (int l = 0; l < c; ++l) {
      for (int j = 0; j < dims_b[1]; ++j) {
        for (int k = 0; k < dims_b[2]; ++k) {
          int idx[4] = {i, j, k, l};
          printf("%f ", bottom_diff->at(idx));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }


  delete bottom, top, bottom_diff, top_diff;

}

template <class Dtype>
__global__ void init_bottom(Tensor<Dtype>* bottom) {
  printf("Initialize bottom tensor...\n");
  const size_t* dims = bottom->GetDims();
  for (int i = 0; i < int(dims[0]); ++i) {
    for (int l = 0; l < int(dims[3]); ++l) {
      for (int j = 0; j < int(dims[1]); ++j) {
        for (int k = 0; k < int(dims[2]); ++k) {
          int idx[4] = {i, j, k, l};
          bottom->at(idx) = (float)(i+j+k+l)/255;
          printf("%f ", bottom->at(idx));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}

template <class Dtype>
__global__ void show_top(Tensor<Dtype>* top) {
  printf("Show top tensor...\n");
  const size_t* dims = top->GetDims();
  for (int i = 0; i < int(dims[0]); ++i) {
    for (int l = 0; l < int(dims[3]); ++l) {
      for (int j = 0; j < int(dims[1]); ++j) {
        for (int k = 0; k < int(dims[2]); ++k) {
          int idx[4] = {i, j, k, l};
          printf("%f ", top->at(idx));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}

template <class Dtype>
__global__ void init_top_diff(Tensor<Dtype>* top_diff) {
  printf("Initialize top diff...\n");
  const size_t* dims = top_diff->GetDims();
  for (int i = 0; i < int(dims[0]); ++i) {
    for (int l = 0; l < int(dims[3]); ++l) {
      for (int j = 0; j < int(dims[1]); ++j) {
        for (int k = 0; k < int(dims[2]); ++k) {
          int idx[4] = {i, j, k, l};
          top_diff->at(idx) = float (i+j+k+l);
          printf("%f ", top_diff->at(idx));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}

template <class Dtype>
__global__ void show_bottom_diff(Tensor<Dtype>* bottom_diff) {
  printf("Show bottom diff...\n");
  const size_t* dims = bottom_diff->GetDims();
  for (int i = 0; i < int(dims[0]); ++i) {
    for (int l = 0; l < int(dims[3]); ++l) {
      for (int j = 0; j < int(dims[1]); ++j) {
        for (int k = 0; k < int(dims[2]); ++k) {
          int idx[4] = {i, j, k, l};
          printf("%f ", bottom_diff->at(idx));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}

void test_pooling_gpu() {
  printf("Start testing pooling layer GPU\n");
  size_t b = 2;
  size_t h = 4;
  size_t w = 4;
  size_t c = 2;

  Session* session = Session::GetNewSession();
  session->gpu = true;
  session->batch_size = b;
 
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  Pooling<float> pooling_layer(2, MAX, 2);

  size_t dims_b[4] = {b, h, w, c};
  Tensor<float>* bottom = Tensor<float>::CreateTensorGPU(dims_b);
  init_bottom<<<1,1>>>(bottom);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  size_t dims_t[4] = {b,0,0,c};
  pooling_layer.GetTopsDims({dims_b}, {dims_t});
  Tensor<float>* top = Tensor<float>::CreateTensorGPU(dims_t);

  pooling_layer.Forward({bottom}, {top});
  printf("Done forward.\n");

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  show_top<<<1,1>>>(top);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  Tensor<float>* top_diff = Tensor<float>::CreateTensorGPU(dims_t);
  init_top_diff<<<1,1>>>(top_diff);

  Tensor<float>* bottom_diff = Tensor<float>::CreateTensorGPU(dims_b);

  pooling_layer.Backward({top},{top_diff},{bottom},{bottom_diff});
  printf("Done backward.\n");

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  show_bottom_diff<<<1,1>>>(bottom_diff);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);

  cudaFree(bottom);
  cudaFree(bottom_diff);
  cudaFree(top);
  cudaFree(top_diff);

}



int main() {
  test_pooling_cpu();
  test_pooling_gpu();
}



