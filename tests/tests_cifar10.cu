#include <stdio.h>
#include <assert.h>
#include "basics/tensor.cu"
#include "basics/session.hpp"
#include "layers/data.cu"
#include "layers/softmax.cu"
#include "layers/cross_entropy_loss.cu"
#include "layers/pooling.cu"
#include "layers/conv2d.cu"
#include "layers/relu.cu"
#include "layers/fc.cu"
#include "tmp/bitmap_image.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils/helper_cuda.h"
#include "utils/utils.cu"
#include "utils/load_model.hpp"

void test_lenet_gpu() {
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);
  show_mem(cudaStatus);
  startTimer();

  Session* session = Session::GetNewSession();
  session->gpu = true;
  session->batch_size = 2;
  size_t batch_size = session->batch_size;


  Data<float> data_layer(batch_size, "datasets/cifar10/train.txt");
  // vector<size_t*> data_tops_dims;
  size_t data_tops_dims0[4];
  size_t data_tops_dims1[4];
  data_layer.GetTopsDims({}, {data_tops_dims0, data_tops_dims1});
  std::vector<Tensor<float>*> data_tops;
  data_tops.push_back(Tensor<float>::CreateTensorGPU(data_tops_dims0));
  data_tops.push_back(Tensor<float>::CreateTensorGPU(data_tops_dims1));

  Conv2D<float> conv1(5,5,3,20,1, new GaussianKernelInitializer<float>(0.1), VALID);
  size_t conv1_top_dims[4];
  conv1.GetTopsDims({data_tops_dims0}, {conv1_top_dims});
  Tensor<float> * conv1_top = Tensor<float>::CreateTensorGPU(conv1_top_dims);

  Pooling<float> pool1(2);
  size_t pool1_top_dims[4];
  pool1.GetTopsDims({conv1_top_dims}, {pool1_top_dims});
  Tensor<float> * pool1_top = Tensor<float>::CreateTensorGPU(pool1_top_dims);
  
  Conv2D<float> conv2(5,5,20,50,1, new GaussianKernelInitializer<float>(0.1), VALID);
  size_t conv2_top_dims[4];
  conv2.GetTopsDims({pool1_top_dims}, {conv2_top_dims});
  printf("pool1 top dims: %d %d %d %d \n", (int)pool1_top_dims[0], (int)pool1_top_dims[1], (int)pool1_top_dims[2], (int)pool1_top_dims[3]);
  printf("conv2 top dims: %d %d %d %d \n", (int)conv2_top_dims[0], (int)conv2_top_dims[1], (int)conv2_top_dims[2], (int)conv2_top_dims[3]);
  Tensor<float> * conv2_top = Tensor<float>::CreateTensorGPU(conv2_top_dims);

  Pooling<float> pool2(2);
  size_t pool2_top_dims[4];
  pool2.GetTopsDims({conv2_top_dims}, {pool2_top_dims});
  Tensor<float> * pool2_top = Tensor<float>::CreateTensorGPU(pool2_top_dims);

  size_t to_fc3_dims[4];
  to_fc3_dims[0] = pool2_top_dims[0];
  to_fc3_dims[1] = 1;
  to_fc3_dims[2] = 1;
  to_fc3_dims[3] = pool2_top_dims[1]*pool2_top_dims[2]*pool2_top_dims[3];
  FC<float> fc3(to_fc3_dims[3],500);
  
  size_t fc3_top_dims[4];

  fc3.GetTopsDims({to_fc3_dims}, {fc3_top_dims});
  printf("pool2 top dims: %d %d %d %d \n", pool2_top_dims[0], pool2_top_dims[1], pool2_top_dims[2], pool2_top_dims[3]);
  printf("fc3 top dims: %d %d %d %d \n", fc3_top_dims[0], fc3_top_dims[1], fc3_top_dims[2], fc3_top_dims[3]);
  Tensor<float> * fc3_top = Tensor<float>::CreateTensorGPU(fc3_top_dims);

  Relu<float> relu3;
  size_t relu3_top_dims[4];
  relu3.GetTopsDims({fc3_top_dims}, {relu3_top_dims});
  Tensor<float> * relu3_top = Tensor<float>::CreateTensorGPU(relu3_top_dims);

  FC<float> fc4(500, 10);
  size_t fc4_top_dims[4];
  fc4.GetTopsDims({relu3_top_dims}, {fc4_top_dims});
  Tensor<float> * fc4_top = Tensor<float>::CreateTensorGPU(fc4_top_dims);

  Softmax<float> softmax;
  size_t sm_top_dims[4];
  softmax.GetTopsDims({fc4_top_dims}, {sm_top_dims});
  Tensor<float> * sm_top = Tensor<float>::CreateTensorGPU(sm_top_dims);

  printf("network finished setup: %3.1f ms \n", stopTimer());
  show_mem(cudaStatus);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
 

  printf("Loading weights ...\n");

  std::string model_path = "models/cifar10/model.txt";
  std::ifstream file(model_path);

  size_t conv1_w_dims[4] = {5,5,3,20};
  Tensor<float>* conv1_w = Tensor<float>::CreateTensorCPU(conv1_w_dims);
  load_to_conv<float>(conv1_w, file);

  size_t conv1_b_dims[4] = {1,1,1,20};
  Tensor<float>* conv1_b = Tensor<float>::CreateTensorCPU(conv1_b_dims);
  load_to_bias<float>(conv1_b, file);

  size_t conv2_w_dims[4] = {5,5,20,50};
  Tensor<float>* conv2_w = Tensor<float>::CreateTensorCPU(conv2_w_dims);
  load_to_conv<float>(conv2_w, file);

  size_t conv2_b_dims[4] = {1,1,1,50};
  Tensor<float>* conv2_b = Tensor<float>::CreateTensorCPU(conv2_b_dims);
  load_to_bias<float>(conv2_b, file);
 
  size_t fc3_w_dims[4] = {1,1,500,1250};
  Tensor<float>* fc3_w = Tensor<float>::CreateTensorCPU(fc3_w_dims);
  load_to_fc<float>(fc3_w, file);

  size_t fc3_b_dims[4] = {1,1,1,500};
  Tensor<float>* fc3_b = Tensor<float>::CreateTensorCPU(fc3_b_dims);
  load_to_bias<float>(fc3_b, file);

  size_t fc4_w_dims[4] = {1,1,10,500};
  Tensor<float>* fc4_w = Tensor<float>::CreateTensorCPU(fc4_w_dims);
  load_to_fc<float>(fc4_w, file);

  size_t fc4_b_dims[4] = {1,1,1,10};
  Tensor<float>* fc4_b = Tensor<float>::CreateTensorCPU(fc4_b_dims);
  load_to_bias<float>(fc4_b, file);


  


  startTimer();
  data_layer.Forward(std::vector<Tensor<float>*> (), data_tops);
  printf("data forward: %3.1f ms \n", stopTimer()); startTimer();
  conv1.Forward({data_tops[0]}, {conv1_top});
  printf("conv1 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool1.Forward({conv1_top}, {pool1_top});
  printf("pool1 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv2.Forward({pool1_top}, {conv2_top});
  printf("conv2 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool2.Forward({conv2_top}, {pool2_top});
  printf("pool2 forward: %3.1f ms \n", stopTimer()); startTimer();
  // flatten the tensor
  Tensor<float>::ReshapeTensorGPU(pool2_top, to_fc3_dims);
  fc3.Forward({pool2_top}, {fc3_top});
  printf("fc3 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu3.Forward({fc3_top}, {relu3_top});
  printf("relu3 forward: %3.1f ms \n", stopTimer()); startTimer();
  fc4.Forward({relu3_top}, {fc4_top});
  printf("fc4 forward: %3.1f ms \n", stopTimer()); startTimer();
  softmax.Forward({fc4_top}, {sm_top});
  printf("softmax forward: %3.1f ms \n", stopTimer()); startTimer();
  show_mem(cudaStatus);






  startTimer();
  data_layer.Forward(std::vector<Tensor<float>*> (), data_tops);
  conv1.Forward({data_tops[0]}, {conv1_top});
  pool1.Forward({conv1_top}, {pool1_top});
  conv2.Forward({pool1_top}, {conv2_top});
  pool2.Forward({conv2_top}, {pool2_top});
  fc3.Forward({pool2_top}, {fc3_top});
  relu3.Forward({fc3_top}, {relu3_top});
  fc4.Forward({relu3_top}, {fc4_top});
  softmax.Forward({fc4_top}, {sm_top});
  printf("finished forward: %3.1f ms \n", stopTimer());
  show_mem(cudaStatus);


/*
  Tensor<float> * output_cpu = Tensor<float>::TensorGPUtoCPU(sm_top);
  Tensor<float> * fc4_cpu = Tensor<float>::TensorGPUtoCPU(fc4_top);


  // printf("%f \n", sm_top->at(0,0,0,0));
  for(int i = 0; i < sm_top_dims[0]; i++) {
  	for(int j = 0; j < sm_top_dims[3]; j++) {
  	  printf("%f ", output_cpu->at(i, 0, 0, j));
  	}
  	printf("\n");
  }

*//*  for(int i = 0; i < fc4_top_dims[0]; i++) {
  	for(int j = 0; j < fc4_top_dims[3]; j++) {
  	  printf("%f ", fc4_cpu->at(i, 0, 0, j));
  	}
  	printf("\n");
  }
*/


  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

}





int main() {
  test_lenet_gpu();
}
