
#include "layers/data.cu"
#include <vector>
#include <assert.h>

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

void test_data_cpu() {
  printf("Begin test data layer CPU\n");
  Session* session = Session::GetNewSession();
  session->gpu = false;

  Data<float> data_layer(2, "/home/jyh/github/Teaism/tmp/test/img_list.txt");

  printf("Frist image: %s\n", data_layer.img_list[0].c_str());
  printf("First label: %d\n", data_layer.lab_list[0]);
  printf("Number of data: %d\n", data_layer.lab_list.size());
  printf("Image width: %d\n", data_layer.img_w);
  printf("Image height: %d\n", data_layer.img_h);
  assert(data_layer.img_list.size() == data_layer.lab_list.size());

  std::vector<Tensor<float>*> top;
  size_t dims_i[4] = {2, data_layer.img_h, data_layer.img_w, 3};
  top.push_back(Tensor<float>::CreateTensorCPU(dims_i));
  size_t dims_l[4] = {2, 1, 1, 1};
  top.push_back(Tensor<float>::CreateTensorCPU(dims_l));

  std::vector<Tensor<float>*> bottom;

  data_layer.Forward(bottom, top);
  
  printf("%f\n", top[1]->at(0,0,0,0));
  printf("%f\n", top[1]->at(1,0,0,0));
  printf("%f\n", top[0]->at(0,1020,531,0));
  printf("%f\n", top[0]->at(1,1020,531,0));
  printf("%f\n", top[0]->at(0,1001,555,1));
  printf("%f\n", top[0]->at(1,1001,555,1));
  printf("%f\n", top[0]->at(0,1000,500,2));
  printf("%f\n", top[0]->at(1,1000,500,2));
}


void test_data_gpu() {
  printf("Begin test data layer GPU\n");
  Session* session = Session::GetNewSession();
  session->gpu = true;

  Data<float> data_layer(2, "/home/jyh/github/Teaism/tmp/test/img_list.txt");

  printf("Frist image: %s\n", data_layer.img_list[0].c_str());
  printf("First label: %d\n", data_layer.lab_list[0]);
  printf("Number of data: %d\n", data_layer.lab_list.size());
  printf("Image width: %d\n", data_layer.img_w);
  printf("Image height: %d\n", data_layer.img_h);
  assert(data_layer.img_list.size() == data_layer.lab_list.size());

  std::vector<Tensor<float>*> top;
  size_t dims_i[4] = {2, data_layer.img_h, data_layer.img_w, 3};
  top.push_back(Tensor<float>::CreateTensorGPU(dims_i));
  size_t dims_l[4] = {2, 1, 1, 1};
  top.push_back(Tensor<float>::CreateTensorGPU(dims_l));

  std::vector<Tensor<float>*> bottom;

  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  data_layer.Forward(bottom, top);
  
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  show_top_label<<<1,1>>>(top[1]);
  show_top<<<1,1>>>(top[0]);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  cudaStatus = cudaDeviceSynchronize();
  checkCudaErrors(cudaStatus);
}


int main() {
  test_data_cpu();
  test_data_gpu();
}
