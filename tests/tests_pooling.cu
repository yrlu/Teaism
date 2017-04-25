#include <stdio.h>
#include "basics/tensor.cu"
#include <assert.h>
#include <cmath>
#include "basics/session.hpp"
#include "layers/pooling.cu"
#include "tmp/bitmap_image.hpp"


void test_pooling_cpu() {
  printf("Example code for pooling layer cpu\n");
  size_t h = 20;
  size_t w = 20;

  Session* session = Session::GetNewSession();
  session->gpu = false;
 
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Pooling<float> * pooling_layer = new Pooling<float>(3, MAX, 2);

  const char* OUTPUT_BMP_PATH = "./tmp/test/pooling_out.bmp";
  size_t b_dims[4] = {1, h, w, 1};
  Tensor<float>* bottom = Tensor<float>::CreateTensorCPU(b_dims);
  size_t t_dims[4] = {0,0,0,0};
  pooling_layer->GetTopsDims({b_dims}, {t_dims});
  printf("%d %d %d %d \n", t_dims[0], t_dims[1], t_dims[2], t_dims[3]);
  Tensor<float>* top = Tensor<float>::CreateTensorCPU(t_dims);

  for(int i = 0; i < h; i++) {
    for(int j = 0; j < w; j++) {
      int b_idx[4] = {0, i, j, 0};
      bottom->at(b_idx) = (float) ((i+j) % 255);
    }
  }
  pooling_layer->Forward({bottom}, {top});

  bitmap_image img(t_dims[1], t_dims[2]);
  for (int i = 0; i < t_dims[1]; i++) {
    for (int j = 0; j < t_dims[2]; j++) {
      unsigned val = (unsigned) top->at(0, i, j, 0);
      img.set_pixel(j, i, val, val, val);
      printf("%f ", top->at(0, i, j, 0));
    }
    printf("\n");
  }
  img.save_image(OUTPUT_BMP_PATH);
  delete pooling_layer;
}



__global__ void init_bottom(Tensor<float> * bottom) {
  for(int i = 0; i < bottom->GetDims()[1]; i++) {
    for(int j = 0; j < bottom->GetDims()[2]; j++) {
      int b_idx[4] = {0, i, j, 0};
      bottom->at(b_idx) = (float) ((i+j) % 255);
    }
  }
}

__global__ void show_top(Tensor<float>* top) {
  size_t h = top->GetDims()[1];
  size_t w = top->GetDims()[2];
  printf("dims: %d %d %d %d \n", (int)top->GetDims()[0], (int)top->GetDims()[1], (int)top->GetDims()[2], (int)top->GetDims()[3]);
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      printf("%f ", top->at(0, i, j, 0));
    }
    printf("\n");
  } 
  printf("%d \n", top->GetDataPtr());
}



void test_pooling_gpu() {
  printf("Example code for pooling layer gpu\n");
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  size_t h = 20;
  size_t w = 20;

  Session* session = Session::GetNewSession();
  session->gpu = true;
 
  Pooling<float> * pooling_layer = new Pooling<float>(3, MAX, 2);
  const char* OUTPUT_BMP_PATH = "./tmp/test/pooling_out_gpu.bmp";
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
 

 
  size_t b_dims[4] = {1, h, w, 1};
  Tensor<float>* bottom = Tensor<float>::CreateTensorGPU(b_dims);
  
  size_t t_dims[4];
  pooling_layer->GetTopsDims({b_dims}, {t_dims});
  printf("t_dims: %d %d %d %d \n", t_dims[0], t_dims[1], t_dims[2], t_dims[3]);
  Tensor<float>* top = Tensor<float>::CreateTensorGPU(t_dims);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  init_bottom<<<1,1>>>(bottom);
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
 
  // show_top<<<1,1>>>(bottom);  
  pooling_layer->Forward({bottom}, {top});
  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  show_top<<<1,1>>>(top);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);


  Tensor<float> * top_cpu = Tensor<float>::TensorGPUtoCPU(top);
  cudaStatus = cudaGetLastError();
  
  checkCudaErrors(cudaStatus);
  
  bitmap_image img(t_dims[1], t_dims[2]); 
  for (int i = 0; i < t_dims[1]; i++) {
    for (int j = 0; j < t_dims[2]; j++) {
      unsigned val = (unsigned) top_cpu->at(0, i, j, 0);
      img.set_pixel(j, i, val, val, val);
    }
  }
  img.save_image(OUTPUT_BMP_PATH);
  delete pooling_layer;
  delete top_cpu;
  cudaFree(bottom);
  cudaFree(top);
}


int main() {
  test_pooling_cpu();
  test_pooling_gpu();
}



