#include <iostream>
#include "layers/conv2d.cu"
#include "basics/tensor.cu"
#include "basics/session.hpp"
#include "tmp/bitmap_image.hpp"
#include "initializers/gaussian_kernel_initializer.cu"
#include <assert.h>
#include <cmath>
#include <stdio.h>

void test_session() {
  std::cout<< "Testing Session .."<<std::endl;
  Session* session = Session::GetNewSession();
  session->gpu = true;
  std::cout<< "use gpu: "<< session->gpu <<std::endl;
}



__device__ Tensor<float>* bottom = NULL; 
__device__ Tensor<float>* top = NULL;
__device__ GaussianKernelInitializer<float>* gaussian_initializer = NULL; 



class Dummy {
public:
  __device__ Dummy(int n):num(n) {

  }

  __device__ int get_num() {
    return num;
  }

private:
  const int num;
};



__device__ Dummy* dummy = NULL;


__global__ void init_dummy() {
  dummy = new Dummy(10000);
  printf("afdasf");
  printf("%d", dummy->get_num());
}

void test_dummy() {
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);
  init_dummy<<<1, 1>>>();
  // use 48KB for shared memory, and 16KB for L1D$
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
}


// __global__ void init_variables() {
//   size_t h = 400;
//   size_t w = 600;
//   bottom = new Tensor<float>({1, h, w, 1});
//   top = new Tensor<float>({1, h/2, w/2, 1});
//   gaussian_initializer = new GaussianKernelInitializer<float>(5.0);
// }

// void test_conv_layer() {
//   std::cout<< "Testing Conv2D .."<<std::endl;
//   size_t h = 400;
//   size_t w = 600;

//   init_variables<<<1, 1>>>();

//   Session* session = Session::GetNewSession();
//   session->gpu = false;

//   // inputs: filter_height, filter_width, in_channels, out_channels, stride
//   Conv2D<float>* conv_layer = new Conv2D<float>(5, 5, 1, 1, 2, gaussian_initializer);
  
//   const char* OUTPUT_BMP_PATH = "./tmp/test/out.bmp";

//   for (int i = 0; i < h; i++) {
//     for (int j = 0; j < w; j++) {
//       bottom->at({0, i, j, 0}) = (float) (rand() % 255);
//     }
//   }
//   // (n, hei, wid, channel)
//   conv_layer->Forward(bottom, top);

//   bitmap_image img(w/2, h/2);
//   for (int i = 0; i < h/2; i++) {
//     for (int j = 0; j < w/2; j++) {
//       unsigned val = (unsigned) top->at({0, i, j, 0});
//       img.set_pixel(j, i, val, val, val);
//     }
//   }
  
//   img.save_image(OUTPUT_BMP_PATH);
//   delete conv_layer;
// }

// void test_tensor() {
//   std::cout<< "Testing Tensor .."<<std::endl;
//   // inputs: tensor dimensions
//   Tensor<float>* tensor = new Tensor<float>({3,3,3});
//   assert(tensor->GetIdx({2,2,2})==26);
//   assert(tensor->GetIdx({1,2,2})==17);
//   assert(tensor->GetIdx({2,1,2})==23);
//   assert(tensor->GetIdx({2,2,1})==25);
//   delete tensor;
// }

// void test_gaussian_kernel() {
//   std::cout<< "Testing gaussian kernel initializer .. "<<std::endl;  
//   std::cout<<Session::GetNewSession()->gpu<<std::endl;
//   Tensor<float>W = Tensor<float>({5,5,1,1});
//   Tensor<float>b = Tensor<float>({1});
//   GaussianKernelInitializer<float>(5.0).Initialize(&W, &b);
//   double sum = 0.0;
//   for (int i = 0; i < W.GetDims()[0]; i++) {
//     for (int j = 0; j < W.GetDims()[1]; j++) {
//       sum += W.at({i, j, 0, 0});
//       std::cout<<W.at({i, j, 0, 0})<<"\t";
//     }
//     std::cout<<std::endl;
//   }
//   assert(std::abs(sum-1.0)<0.00001);
// }


int main(void) {
  test_dummy();
  // test_conv_layer();
  // test_tensor();  
  // test_session();
  // test_gaussian_kernel();
}