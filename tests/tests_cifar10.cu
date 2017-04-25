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




void print_W(Tensor<float>* output) {
  for(int o = 0; o < output->GetDims()[3]; o++) { // out_channels
    for(int i = 0; i < output->GetDims()[2]; i++) { // in_channels
      for(int w = 0; w < output->GetDims()[1]; w++) { // kernel wid
        for(int h = 0; h < output->GetDims()[0]; h++) { // kernel hei
          std::cout<< output->at(h, w, i, o)<< " ";
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}

__global__ void print_W_gpu(Tensor<float> * output) {
  for(int o = 0; o < output->GetDims()[3]; o++) { // out_channels
    for(int i = 0; i < output->GetDims()[2]; i++) { // in_channels
      for(int w = 0; w < output->GetDims()[1]; w++) { // kernel wid
        for(int h = 0; h < output->GetDims()[0]; h++) { // kernel hei
          printf("%f ", output->at(h, w, i, o));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  } 
}

void print_conv_top(Tensor<float> * conv_top) {
  // b, h, w, o
  for(int b = 0; b < conv_top->GetDims()[0]; b++) {
    for(int o = 0; o < conv_top->GetDims()[3]; o++) {
      for(int h = 0; h < conv_top->GetDims()[1]; h++) {
        for(int w = 0; w < conv_top->GetDims()[2]; w++) {
          std::cout<<conv_top->at(b, h, w, o) << " ";
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
}


__global__ void print_conv_top_gpu(Tensor<float> * conv_top) {
  for(int b = 0; b < conv_top->GetDims()[0]; b++) {
    for(int o = 0; o < conv_top->GetDims()[3]; o++) {
      for(int h = 0; h < conv_top->GetDims()[1]; h++) {
        for(int w = 0; w < conv_top->GetDims()[2]; w++) {
          printf("%f ", conv_top->at(b, h, w, o));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
  printf("\n");
}

void print_fc(Tensor<float> * fc_top) {
  // b, o
  assert(fc_top->GetDims()[1]==1);
  assert(fc_top->GetDims()[2]==1);
  for(int b = 0; b < fc_top->GetDims()[0]; b++) {
    for(int o = 0; o < fc_top->GetDims()[3]; o++) {
      std::cout<<fc_top->at(b, 0, 0, o) << " ";
    }
    printf("\n");
  }
  printf("\n");
}

void test_lenet_cpu() {
  startTimer();
  Session* session = Session::GetNewSession();
  session->gpu = false;
  session->batch_size = 1;
  size_t batch_size = session->batch_size;


  Data<float> data_layer(batch_size, "datasets/cifar10/train.txt");
  // vector<size_t*> data_tops_dims;
  size_t data_tops_dims0[4];
  size_t data_tops_dims1[4];
  data_layer.GetTopsDims({}, {data_tops_dims0, data_tops_dims1});
  std::vector<Tensor<float>*> data_tops;
  data_tops.push_back(Tensor<float>::CreateTensorCPU(data_tops_dims0));
  data_tops.push_back(Tensor<float>::CreateTensorCPU(data_tops_dims1));

  Conv2D<float> conv1(5,5,3,32,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv1_top_dims[4];
  conv1.GetTopsDims({data_tops_dims0}, {conv1_top_dims});
  Tensor<float> * conv1_top = Tensor<float>::CreateTensorCPU(conv1_top_dims);

  Pooling<float> pool1(2, MAX, 2);
  size_t pool1_top_dims[4];
  pool1.GetTopsDims({conv1_top_dims}, {pool1_top_dims});
  Tensor<float> * pool1_top = Tensor<float>::CreateTensorCPU(pool1_top_dims);
  
  Relu<float> relu1;
  size_t relu1_top_dims[4];
  relu1.GetTopsDims({pool1_top_dims}, {relu1_top_dims});
  Tensor<float> * relu1_top = Tensor<float>::CreateTensorCPU(relu1_top_dims);

  Conv2D<float> conv2(5,5,32,32,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv2_top_dims[4];
  conv2.GetTopsDims({relu1_top_dims}, {conv2_top_dims});
  Tensor<float> * conv2_top = Tensor<float>::CreateTensorCPU(conv2_top_dims);

  Pooling<float> pool2(2, MAX, 2);
  size_t pool2_top_dims[4];
  pool2.GetTopsDims({conv2_top_dims}, {pool2_top_dims});
  Tensor<float> * pool2_top = Tensor<float>::CreateTensorCPU(pool2_top_dims);

  Relu<float> relu2;
  size_t relu2_top_dims[4];
  relu2.GetTopsDims({pool2_top_dims}, {relu2_top_dims});
  Tensor<float> * relu2_top = Tensor<float>::CreateTensorCPU(relu2_top_dims);

  Conv2D<float> conv3(5,5,32,64,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv3_top_dims[4];
  conv3.GetTopsDims({relu2_top_dims}, {conv3_top_dims});
  Tensor<float> * conv3_top = Tensor<float>::CreateTensorCPU(conv3_top_dims);

  Pooling<float> pool3(2, MAX, 2);
  size_t pool3_top_dims[4];
  pool3.GetTopsDims({conv3_top_dims}, {pool3_top_dims});
  Tensor<float> * pool3_top = Tensor<float>::CreateTensorCPU(pool3_top_dims);

  Relu<float> relu3;
  size_t relu3_top_dims[4];
  relu3.GetTopsDims({pool3_top_dims}, {relu3_top_dims});
  Tensor<float> * relu3_top = Tensor<float>::CreateTensorCPU(relu3_top_dims);

  size_t to_fc4_dims[4];
  to_fc4_dims[0] = relu3_top_dims[0];
  to_fc4_dims[1] = 1;
  to_fc4_dims[2] = 1;
  to_fc4_dims[3] = relu3_top_dims[1]*relu3_top_dims[2]*relu3_top_dims[3];
  FC<float> fc4(to_fc4_dims[3],64);
  printf("relu3 top dims: %d %d %d %d \n", relu3_top_dims[0], relu3_top_dims[1], relu3_top_dims[2], relu3_top_dims[3]);
  printf("to fc4 dims: %d %d %d %d \n", to_fc4_dims[0], to_fc4_dims[1], to_fc4_dims[2], to_fc4_dims[3]);
  size_t fc4_top_dims[4];
  fc4.GetTopsDims({to_fc4_dims}, {fc4_top_dims});
  Tensor<float> * fc4_top = Tensor<float>::CreateTensorCPU(fc4_top_dims);

  FC<float> fc5(64, 10);
  size_t fc5_top_dims[4];
  fc5.GetTopsDims({fc4_top_dims}, {fc5_top_dims});
  Tensor<float> * fc5_top = Tensor<float>::CreateTensorCPU(fc5_top_dims);

  Softmax<float> softmax;
  size_t sm_top_dims[4];
  softmax.GetTopsDims({fc5_top_dims}, {sm_top_dims});
  Tensor<float> * sm_top = Tensor<float>::CreateTensorCPU(sm_top_dims);

  printf("network finished setup: %3.1f ms \n", stopTimer());
 

  printf("Loading weights ...\n");

  std::string model_path = "models/cifar10/model.txt";
  std::ifstream file(model_path);

  size_t conv1_w_dims[4] = {5,5,3,32};
  Tensor<float>* conv1_w = Tensor<float>::CreateTensorCPU(conv1_w_dims);
  load_to_conv<float>(conv1_w, file);
  conv1.W_->SetDataPtr(conv1_w->GetDataPtr());

  size_t conv1_b_dims[4] = {1,1,1,32};
  Tensor<float>* conv1_b = Tensor<float>::CreateTensorCPU(conv1_b_dims);
  load_to_bias<float>(conv1_b, file);
  conv1.b_->SetDataPtr(conv1_b->GetDataPtr());

  size_t conv2_w_dims[4] = {5,5,32,32};
  Tensor<float>* conv2_w = Tensor<float>::CreateTensorCPU(conv2_w_dims);
  load_to_conv<float>(conv2_w, file);
  conv2.W_->SetDataPtr(conv2_w->GetDataPtr());

  size_t conv2_b_dims[4] = {1,1,1,32};
  Tensor<float>* conv2_b = Tensor<float>::CreateTensorCPU(conv2_b_dims);
  load_to_bias<float>(conv2_b, file);
  conv2.b_->SetDataPtr(conv2_b->GetDataPtr());

  size_t conv3_w_dims[4] = {5,5,32,64};
  Tensor<float>* conv3_w = Tensor<float>::CreateTensorCPU(conv3_w_dims);
  load_to_conv<float>(conv3_w, file);
  conv3.W_->SetDataPtr(conv3_w->GetDataPtr());

  size_t conv3_b_dims[4] = {1,1,1,64};
  Tensor<float>* conv3_b = Tensor<float>::CreateTensorCPU(conv3_b_dims);
  load_to_bias<float>(conv3_b, file);
  conv3.b_->SetDataPtr(conv3_b->GetDataPtr());

  size_t fc4_w_dims[4] = {1,1,64,1024};
  Tensor<float>* fc4_w = Tensor<float>::CreateTensorCPU(fc4_w_dims);
  load_to_fc<float>(fc4_w, file);
  fc4.W_->SetDataPtr(fc4_w->GetDataPtr());


  size_t fc4_b_dims[4] = {1,1,1,64};
  Tensor<float>* fc4_b = Tensor<float>::CreateTensorCPU(fc4_b_dims);
  load_to_bias<float>(fc4_b, file);
  fc4.b_->SetDataPtr(fc4_b->GetDataPtr());

  size_t fc5_w_dims[4] = {1,1,10,64};
  Tensor<float>* fc5_w = Tensor<float>::CreateTensorCPU(fc5_w_dims);
  load_to_fc<float>(fc5_w, file);
  fc5.W_->SetDataPtr(fc5_w->GetDataPtr());

  size_t fc5_b_dims[4] = {1,1,1,10};
  Tensor<float>* fc5_b = Tensor<float>::CreateTensorCPU(fc5_b_dims);
  load_to_bias<float>(fc5_b, file);
  fc5.b_->SetDataPtr(fc5_b->GetDataPtr());


  startTimer();
  data_layer.Forward(std::vector<Tensor<float>*> (), data_tops);
  printf("data forward: %3.1f ms \n", stopTimer()); startTimer();
  conv1.Forward({data_tops[0]}, {conv1_top});
  printf("conv1 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool1.Forward({conv1_top}, {pool1_top});
  printf("pool1 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu1.Forward({pool1_top}, {relu1_top});
  printf("relu1 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv2.Forward({relu1_top}, {conv2_top});
  printf("conv2 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool2.Forward({conv2_top}, {pool2_top});
  printf("pool2 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu2.Forward({pool2_top}, {relu2_top});
  printf("relu2 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv3.Forward({relu2_top}, {conv3_top});
  printf("conv3 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool3.Forward({conv3_top}, {pool3_top});
  printf("pool3 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu3.Forward({pool3_top}, {relu3_top});
  printf("relu3 forward: %3.1f ms \n", stopTimer()); startTimer();
  // flatten the tensor

  size_t relu3_top_dims_reshaped[4] = {relu3_top_dims[0], relu3_top_dims[3], relu3_top_dims[1], relu3_top_dims[2]};
  Tensor<float> * reshaped_relu3_top = Tensor<float>::CreateTensorCPU(relu3_top_dims_reshaped);
  for(int b = 0; b < relu3_top_dims_reshaped[0]; b++) {
    for(int c = 0; c < relu3_top_dims_reshaped[1]; c++) {
      for(int h = 0; h < relu3_top_dims_reshaped[2]; h++) {
        for(int w = 0; w < relu3_top_dims_reshaped[3]; w++) {
          reshaped_relu3_top->at(b, c, h, w) = relu3_top->at(b, h, w, c);
        }
      }
    }
  }

  Tensor<float>::ReshapeTensorCPU(reshaped_relu3_top, to_fc4_dims);
  printf("reshaped relu3 top dims: %d %d %d %d \n", reshaped_relu3_top->GetDims()[0], reshaped_relu3_top->GetDims()[1], reshaped_relu3_top->GetDims()[2], reshaped_relu3_top->GetDims()[3]);
  fc4.Forward({reshaped_relu3_top}, {fc4_top});
  printf("fc4 forward: %3.1f ms \n", stopTimer()); startTimer();
  fc5.Forward({fc4_top}, {fc5_top});
  printf("fc5 forward: %3.1f ms \n", stopTimer()); startTimer();
  softmax.Forward({fc5_top}, {sm_top});
  printf("softmax forward: %3.1f ms \n", stopTimer()); startTimer();

  // Tensor<float> * output_cpu = Tensor<float>::TensorGPUtoCPU(conv1.W_);
  // Tensor<float> * fc4_cpu = Tensor<float>::TensorGPUtoCPU(fc4_top);
  // print_W(output_cpu);
  // Tensor<float> * conv_top = Tensor<float>::TensorGPUtoCPU(conv1_top);
  print_conv_top(conv3_top);
  // print_fc();
}

void test_lenet_gpu() {
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);
  show_mem(cudaStatus);
  startTimer();

  Session* session = Session::GetNewSession();
  session->gpu = true;
  session->batch_size = 1;
  size_t batch_size = session->batch_size;


  Data<float> data_layer(batch_size, "datasets/cifar10/train.txt");
  // vector<size_t*> data_tops_dims;
  size_t data_tops_dims0[4];
  size_t data_tops_dims1[4];
  data_layer.GetTopsDims({}, {data_tops_dims0, data_tops_dims1});
  std::vector<Tensor<float>*> data_tops;
  data_tops.push_back(Tensor<float>::CreateTensorGPU(data_tops_dims0));
  data_tops.push_back(Tensor<float>::CreateTensorGPU(data_tops_dims1));

  Conv2D<float> conv1(5,5,3,32,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv1_top_dims[4];
  conv1.GetTopsDims({data_tops_dims0}, {conv1_top_dims});
  Tensor<float> * conv1_top = Tensor<float>::CreateTensorGPU(conv1_top_dims);

  Pooling<float> pool1(2, MAX, 2);
  size_t pool1_top_dims[4];
  pool1.GetTopsDims({conv1_top_dims}, {pool1_top_dims});
  Tensor<float> * pool1_top = Tensor<float>::CreateTensorGPU(pool1_top_dims);
  
  Relu<float> relu1;
  size_t relu1_top_dims[4];
  relu1.GetTopsDims({pool1_top_dims}, {relu1_top_dims});
  Tensor<float> * relu1_top = Tensor<float>::CreateTensorGPU(relu1_top_dims);

  Conv2D<float> conv2(5,5,32,32,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv2_top_dims[4];
  conv2.GetTopsDims({relu1_top_dims}, {conv2_top_dims});
  Tensor<float> * conv2_top = Tensor<float>::CreateTensorGPU(conv2_top_dims);

  Pooling<float> pool2(2, MAX, 2);
  size_t pool2_top_dims[4];
  pool2.GetTopsDims({conv2_top_dims}, {pool2_top_dims});
  Tensor<float> * pool2_top = Tensor<float>::CreateTensorGPU(pool2_top_dims);

  Relu<float> relu2;
  size_t relu2_top_dims[4];
  relu2.GetTopsDims({pool2_top_dims}, {relu2_top_dims});
  Tensor<float> * relu2_top = Tensor<float>::CreateTensorGPU(relu2_top_dims);

  Conv2D<float> conv3(5,5,32,64,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv3_top_dims[4];
  conv3.GetTopsDims({relu2_top_dims}, {conv3_top_dims});
  Tensor<float> * conv3_top = Tensor<float>::CreateTensorGPU(conv3_top_dims);

  Pooling<float> pool3(2, MAX, 2);
  size_t pool3_top_dims[4];
  pool3.GetTopsDims({conv3_top_dims}, {pool3_top_dims});
  Tensor<float> * pool3_top = Tensor<float>::CreateTensorGPU(pool3_top_dims);

  Relu<float> relu3;
  size_t relu3_top_dims[4];
  relu3.GetTopsDims({pool3_top_dims}, {relu3_top_dims});
  Tensor<float> * relu3_top = Tensor<float>::CreateTensorGPU(relu3_top_dims);

  size_t to_fc4_dims[4];
  to_fc4_dims[0] = relu3_top_dims[0];
  to_fc4_dims[1] = 1;
  to_fc4_dims[2] = 1;
  to_fc4_dims[3] = relu3_top_dims[1]*relu3_top_dims[2]*relu3_top_dims[3];
  FC<float> fc4(to_fc4_dims[3],64);
  size_t fc4_top_dims[4];
  fc4.GetTopsDims({to_fc4_dims}, {fc4_top_dims});
  Tensor<float> * fc4_top = Tensor<float>::CreateTensorGPU(fc4_top_dims);

  FC<float> fc5(64, 10);
  size_t fc5_top_dims[4];
  fc5.GetTopsDims({fc4_top_dims}, {fc5_top_dims});
  Tensor<float> * fc5_top = Tensor<float>::CreateTensorGPU(fc5_top_dims);

  Softmax<float> softmax;
  size_t sm_top_dims[4];
  softmax.GetTopsDims({fc5_top_dims}, {sm_top_dims});
  Tensor<float> * sm_top = Tensor<float>::CreateTensorGPU(sm_top_dims);

  printf("network finished setup: %3.1f ms \n", stopTimer());
  show_mem(cudaStatus);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);


  printf("Loading weights ...\n");

  std::string model_path = "models/cifar10/model.txt";
  std::ifstream file(model_path);

  size_t conv1_w_dims[4] = {5,5,3,32};
  Tensor<float>* conv1_w = Tensor<float>::CreateTensorCPU(conv1_w_dims);
  load_to_conv<float>(conv1_w, file);
  Tensor<float>::DataArrayCPUtoGPU(conv1_w, conv1.W_);

  size_t conv1_b_dims[4] = {1,1,1,32};
  Tensor<float>* conv1_b = Tensor<float>::CreateTensorCPU(conv1_b_dims);
  load_to_bias<float>(conv1_b, file);
  Tensor<float>::DataArrayCPUtoGPU(conv1_b, conv1.b_);

  size_t conv2_w_dims[4] = {5,5,32,32};
  Tensor<float>* conv2_w = Tensor<float>::CreateTensorCPU(conv2_w_dims);
  load_to_conv<float>(conv2_w, file);
  Tensor<float>::DataArrayCPUtoGPU(conv2_w, conv2.W_);

  size_t conv2_b_dims[4] = {1,1,1,32};
  Tensor<float>* conv2_b = Tensor<float>::CreateTensorCPU(conv2_b_dims);
  load_to_bias<float>(conv2_b, file);
  Tensor<float>::DataArrayCPUtoGPU(conv2_b, conv2.b_);

   size_t conv3_w_dims[4] = {5,5,32,64};
  Tensor<float>* conv3_w = Tensor<float>::CreateTensorCPU(conv3_w_dims);
  load_to_conv<float>(conv3_w, file);
  Tensor<float>::DataArrayCPUtoGPU(conv3_w, conv3.W_);

  size_t conv3_b_dims[4] = {1,1,1,64};
  Tensor<float>* conv3_b = Tensor<float>::CreateTensorCPU(conv3_b_dims);
  load_to_bias<float>(conv3_b, file);
  Tensor<float>::DataArrayCPUtoGPU(conv3_b, conv3.b_);

  size_t fc4_w_dims[4] = {1,1,64,1024};
  Tensor<float>* fc4_w = Tensor<float>::CreateTensorCPU(fc4_w_dims);
  load_to_fc<float>(fc4_w, file);
  Tensor<float>::DataArrayCPUtoGPU(fc4_w, fc4.W_);

  size_t fc4_b_dims[4] = {1,1,1,64};
  Tensor<float>* fc4_b = Tensor<float>::CreateTensorCPU(fc4_b_dims);
  load_to_bias<float>(fc4_b, file);
  Tensor<float>::DataArrayCPUtoGPU(fc4_b, fc4.b_);

  size_t fc5_w_dims[4] = {1,1,10,64};
  Tensor<float>* fc5_w = Tensor<float>::CreateTensorCPU(fc5_w_dims);
  load_to_fc<float>(fc5_w, file);
  Tensor<float>::DataArrayCPUtoGPU(fc5_w, fc5.W_);

  size_t fc5_b_dims[4] = {1,1,1,10};
  Tensor<float>* fc5_b = Tensor<float>::CreateTensorCPU(fc5_b_dims);
  load_to_bias<float>(fc5_b, file);
  Tensor<float>::DataArrayCPUtoGPU(fc5_b, fc5.b_);

  


  startTimer();
  data_layer.Forward(std::vector<Tensor<float>*> (), data_tops);
  printf("data forward: %3.1f ms \n", stopTimer()); startTimer();
  conv1.Forward({data_tops[0]}, {conv1_top});
  printf("conv1 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool1.Forward({conv1_top}, {pool1_top});
  printf("pool1 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu1.Forward({pool1_top}, {relu1_top});
  printf("relu1 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv2.Forward({relu1_top}, {conv2_top});
  printf("conv2 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool2.Forward({conv2_top}, {pool2_top});
  printf("pool2 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu2.Forward({pool2_top}, {relu2_top});
  printf("relu2 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv3.Forward({relu2_top}, {conv3_top});
  printf("conv3 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool3.Forward({conv3_top}, {pool3_top});
  printf("pool3 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu3.Forward({pool3_top}, {relu3_top});
  printf("relu3 forward: %3.1f ms \n", stopTimer()); startTimer();
  // flatten the tensor
  Tensor<float>::ReshapeTensorGPU(relu3_top, to_fc4_dims);
  fc4.Forward({relu3_top}, {fc4_top});
  printf("fc4 forward: %3.1f ms \n", stopTimer()); startTimer();
  fc5.Forward({fc4_top}, {fc5_top});
  printf("fc5 forward: %3.1f ms \n", stopTimer()); startTimer();
  softmax.Forward({fc5_top}, {sm_top});
  printf("softmax forward: %3.1f ms \n", stopTimer()); startTimer();
  show_mem(cudaStatus);





/*
  startTimer();
  data_layer.Forward(std::vector<Tensor<float>*> (), data_tops);
  conv1.Forward({data_tops[0]}, {conv1_top});
  pool1.Forward({conv1_top}, {pool1_top});
  relu1.Forward({pool1_top}, {relu1_top});
  conv2.Forward({relu1_top}, {conv2_top});
  pool2.Forward({conv2_top}, {pool2_top});
  relu2.Forward({pool2_top}, {relu2_top});
  conv3.Forward({relu2_top}, {conv3_top});
  pool3.Forward({conv3_top}, {pool3_top});
  relu3.Forward({pool3_top}, {relu3_top});
  fc4.Forward({relu3_top}, {fc4_top});
  fc5.Forward({fc4_top}, {fc5_top});
  softmax.Forward({fc5_top}, {sm_top});
  printf("finished forward: %3.1f ms \n", stopTimer());
  show_mem(cudaStatus);
*/

  // print_W_gpu<<<1,1>>>(conv1.W_);
  // print_W(conv1_w);

  // Tensor<float> * output_cpu = Tensor<float>::TensorGPUtoCPU(conv1.W_);
//  Tensor<float> * fc4_cpu = Tensor<float>::TensorGPUtoCPU(fc4_top);
  // print_W(output_cpu);
  Tensor<float> * conv_top = Tensor<float>::TensorGPUtoCPU(conv3_top);
  print_conv_top(conv_top);
  // print_conv_top_gpu<<<1,1>>>(conv1_top);

/*
  // printf("%f \n", sm_top->at(0,0,0,0));
  for(int b = 0; b < output_cpu->GetDims()[0]; b++) {
    for(int h = 0; h < output_cpu->GetDims()[1]; h++) {
      for(int w = 0; w < output_cpu->GetDims()[2]; w++) {
  	    for(int c = 0; c < output_cpu->GetDims()[3]; c++) {
  	      printf("%f ", output_cpu->at(b, h, w, c));
  	    }
  	    printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
*/
/*  for(int i = 0; i < fc4_top_dims[0]; i++) {
  	for(int j = 0; j < fc4_top_dims[3]; j++) {
  	  printf("%f ", fc4_cpu->at(i, 0, 0, j));
  	}
  	printf("\n");
  }
*/


  // cudaStatus = cudaGetLastError();
  // checkCudaErrors(cudaStatus);

}



int main() {
  test_lenet_gpu();
  // test_lenet_cpu();
}
