#include <stdio.h>
#include "basics/tensor.cu"
#include <assert.h>
#include <cmath>
#include "basics/session.hpp"
#include "layers/relu.cu"


void test_relu_cpu() {
  printf("Example code for relu layer cpu\n");
  size_t b = 2;
  size_t h = 2;
  size_t w = 2;
  size_t c = 3;

  Session* session = Session::GetNewSession();
  session->gpu = false;
 
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Relu<float> * relu = new Relu<float>();

  size_t dims[4] = {b, h, w, c};
  Tensor<float>* top = Tensor<float>::CreateTensorCPU(dims);
  Tensor<float>* top_diff = Tensor<float>::CreateTensorCPU(dims);
  Tensor<float>* bottom = Tensor<float>::CreateTensorCPU(dims);
  Tensor<float>* bottom_diff = Tensor<float>::CreateTensorCPU(dims);

  printf("Initialize bottom: \n");
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        for (int l = 0; l < c; l++) {
          bottom->at(i,j,k,l) = i+j-k-l;
          printf("%f ",bottom->at(i,j,k,l)); }
        printf("\n"); }
      printf("\n"); }
    printf("\n"); }

  relu->Forward({bottom}, {top});

  printf("Show top: \n");
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        for (int l = 0; l < c; l++) {
          printf("%f ",top->at(i,j,k,l)); }
        printf("\n"); }
      printf("\n"); }
    printf("\n"); }


  printf("Initialize top diff: \n");
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        for (int l = 0; l < c; l++) {
          top_diff->at(i,j,k,l) = 0.5;
          printf("%f ",top_diff->at(i,j,k,l)); }
        printf("\n"); }
      printf("\n"); }
    printf("\n"); }

  relu->Backward({top},{top_diff},{bottom},{bottom_diff});

  printf("Show bottom diff: \n");
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        for (int l = 0; l < c; l++) {
          printf("%f ",bottom_diff->at(i,j,k,l)); }
        printf("\n"); }
      printf("\n"); }
    printf("\n"); }


  delete relu;
  delete top, top_diff, bottom, bottom_diff;
}



void test_relu_gpu() {
  printf("Example code for relu layer gpu\n");
  size_t b = 2;
  size_t h = 2;
  size_t w = 2;
  size_t c = 3;

  Session* session = Session::GetNewSession();
  session->gpu = true;
  session->batch_size = b;
 
  // inputs: filter_height, filter_width, in_channels, out_channels, stride
  Relu<float> * relu = new Relu<float>();

  size_t dims[4] = {b, h, w, c};
  Tensor<float>* top = Tensor<float>::CreateTensorGPU(dims);
  Tensor<float>* bottom_diff = Tensor<float>::CreateTensorGPU(dims);

  Tensor<float>* bottom_cpu = Tensor<float>::CreateTensorCPU(dims);
  printf("Initialize bottom: \n");
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        for (int l = 0; l < c; l++) {
          bottom_cpu->at(i,j,k,l) = i+j-k-l;
          printf("%f ",bottom_cpu->at(i,j,k,l)); }
        printf("\n"); }
      printf("\n"); }
    printf("\n"); }
  Tensor<float>* bottom = Tensor<float>::TensorCPUtoGPU(bottom_cpu);

  relu->Forward({bottom}, {top});

  printf("Show top: \n");
  Tensor<float>* top_cpu = Tensor<float>::TensorGPUtoCPU(top);
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        for (int l = 0; l < c; l++) {
          printf("%f ",top_cpu->at(i,j,k,l)); }
        printf("\n"); }
      printf("\n"); }
    printf("\n"); }


  printf("Initialize top diff: \n");
  Tensor<float>* top_diff_cpu = Tensor<float>::CreateTensorCPU(dims);
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        for (int l = 0; l < c; l++) {
          top_diff_cpu->at(i,j,k,l) = 0.5;
          printf("%f ",top_diff_cpu->at(i,j,k,l)); }
        printf("\n"); }
      printf("\n"); }
    printf("\n"); }
  Tensor<float>* top_diff = Tensor<float>::TensorCPUtoGPU(top_diff_cpu);

  relu->Backward({top},{top_diff},{bottom},{bottom_diff});

  printf("Show bottom diff: \n");
  Tensor<float>* bottom_diff_cpu = Tensor<float>::TensorGPUtoCPU(bottom_diff);
  for (int i = 0; i < b; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        for (int l = 0; l < c; l++) {
          printf("%f ",bottom_diff_cpu->at(i,j,k,l)); }
        printf("\n"); }
      printf("\n"); }
    printf("\n"); }


  delete relu;
  cudaFree(top);
  cudaFree(top_diff);
  cudaFree(bottom);
  cudaFree(bottom_diff);
  delete top_cpu, top_diff_cpu, bottom_cpu, bottom_diff_cpu;
}



int main() {
  test_relu_cpu();
  test_relu_gpu();
}



