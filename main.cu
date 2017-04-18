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
  __host__ __device__ Dummy(int n):num(n) {

  }

  __host__ __device__ int get_num() {
    return num;
  }

  __host__ __device__ void get_one() {
  }

private:
  const int num;
};



Dummy* dummy = NULL;


__global__ void init_dummy(Dummy* dummy) {
  dummy->get_one();
}

void test_dummy() {
  dummy = new Dummy(10000);
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);
  init_dummy<<<1, 1>>>(dummy);
  // use 48KB for shared memory, and 16KB for L1D$
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaStatus = cudaDeviceSynchronize();
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


__global__ void test_tensor_gpu(Tensor<float>* tensor) {
  //int idx[4] = {1, 2, 2, 2};
  //printf("%d\n", tensor->GetIdx(idx));
}
void test_tensor() {
  // inputs: tensor dimensions

  cudaError_t cudaStatus = cudaSetDevice(0);
  size_t dims[4] = {3, 3, 3, 3};
  Tensor<float>* tensor_cpu = new Tensor<float>(dims);
  Tensor<float>* tensor_gpu;
  cudaMalloc((void **)&tensor_gpu, sizeof(Tensor<float>));
  cudaMemcpy(tensor_gpu, tensor_cpu, sizeof(Tensor<float>), cudaMemcpyHostToDevice);
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  
  float data_ary_cpu[3*3*3*3];
  float * data_ary;
  cudaMalloc((void**) &data_ary, sizeof(float)*3*3*3*3);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaMemcpy(data_ary, data_ary_cpu, sizeof(float)*81, cudaMemcpyHostToDevice);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaMemcpy(&tensor_gpu->data_array_, &data_ary, sizeof(float*), cudaMemcpyHostToDevice);


  float * data_ary_cpu2 = (float*)malloc(sizeof(float)*81);
  cudaMemcpy(data_ary_cpu2, data_ary, sizeof(float)*81, cudaMemcpyDeviceToHost);
   
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
//  test_tensor_gpu<<<1,1>>>(tensor_gpu);

  delete tensor_cpu;
}

void test_tensor_cpu() {
  size_t dims[4] = {3, 3, 3, 3};
  Tensor<float>* tensor = new Tensor<float>(dims);
  int idx[4] = {1, 2, 2, 2};
  printf("%d\n", tensor->GetIdx(idx));
}

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

  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);


  test_tensor();  
//  test_tensor_cpu();
  // test_session();
  // test_gaussian_kernel();
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);

}
