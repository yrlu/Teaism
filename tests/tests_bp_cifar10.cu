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
#include "utils/bitmap_image.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils/helper_cuda.h"
#include "utils/utils.cu"
#include "utils/load_model.hpp"



__global__ void show_tensor(Tensor<float> * tensor) {
  size_t d1 = tensor->GetDims()[0];
  size_t d2 = tensor->GetDims()[1];
  size_t d3 = tensor->GetDims()[2];
  size_t d4 = tensor->GetDims()[3];

  for(int i = 0; i < d1; i++) {
    for(int l = 0; l < d4; l++) {
      for(int j = 0; j < d2; j++) {
        for(int k = 0; k < d3; k++) {
          printf("%f ", tensor->at(i, j, k, l));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}


void demo_bp_cifar10_gpu() {
  printf("Start training convolutional networks on cifar10\n\n");

  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);
  startTimer();

  Session* session = Session::GetNewSession();
  session->gpu = true;
  session->batch_size = 1;
  size_t batch_size = session->batch_size;


  Data<float> data_layer(batch_size, "datasets/cifar10/train1.txt");
  // vector<size_t*> data_tops_dims;
  size_t data_tops_dims0[4];
  size_t data_tops_dims1[4];
  data_layer.GetTopsDims({}, {data_tops_dims0, data_tops_dims1});
  std::vector<Tensor<float>*> data_tops;
  data_tops.push_back(Tensor<float>::CreateTensorGPU(data_tops_dims0));
  data_tops.push_back(Tensor<float>::CreateTensorGPU(data_tops_dims1));
  Tensor<float> * data_top_diff0 = Tensor<float>::CreateTensorGPU(data_tops_dims0);

  Conv2D<float> conv1(5,5,3,32,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv1_top_dims[4];
  conv1.GetTopsDims({data_tops_dims0}, {conv1_top_dims});
  Tensor<float> * conv1_top = Tensor<float>::CreateTensorGPU(conv1_top_dims);
  Tensor<float> * conv1_top_diff = Tensor<float>::CreateTensorGPU(conv1_top_dims);

  Pooling<float> pool1(2, MAX, 2);
  size_t pool1_top_dims[4];
  pool1.GetTopsDims({conv1_top_dims}, {pool1_top_dims});
  Tensor<float> * pool1_top = Tensor<float>::CreateTensorGPU(pool1_top_dims);
  Tensor<float> * pool1_top_diff = Tensor<float>::CreateTensorGPU(pool1_top_dims);
  
  Relu<float> relu1;
  size_t relu1_top_dims[4];
  relu1.GetTopsDims({pool1_top_dims}, {relu1_top_dims});
  Tensor<float> * relu1_top = Tensor<float>::CreateTensorGPU(relu1_top_dims);
  Tensor<float> * relu1_top_diff = Tensor<float>::CreateTensorGPU(relu1_top_dims);

  Conv2D<float> conv2(5,5,32,32,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv2_top_dims[4];
  conv2.GetTopsDims({relu1_top_dims}, {conv2_top_dims});
  Tensor<float> * conv2_top = Tensor<float>::CreateTensorGPU(conv2_top_dims);
  Tensor<float> * conv2_top_diff = Tensor<float>::CreateTensorGPU(conv2_top_dims);

  Pooling<float> pool2(2, MAX, 2);
  size_t pool2_top_dims[4];
  pool2.GetTopsDims({conv2_top_dims}, {pool2_top_dims});
  Tensor<float> * pool2_top = Tensor<float>::CreateTensorGPU(pool2_top_dims);
  Tensor<float> * pool2_top_diff = Tensor<float>::CreateTensorGPU(pool2_top_dims);

  Relu<float> relu2;
  size_t relu2_top_dims[4];
  relu2.GetTopsDims({pool2_top_dims}, {relu2_top_dims});
  Tensor<float> * relu2_top = Tensor<float>::CreateTensorGPU(relu2_top_dims);
  Tensor<float> * relu2_top_diff = Tensor<float>::CreateTensorGPU(relu2_top_dims);

  Conv2D<float> conv3(5,5,32,64,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv3_top_dims[4];
  conv3.GetTopsDims({relu2_top_dims}, {conv3_top_dims});
  Tensor<float> * conv3_top = Tensor<float>::CreateTensorGPU(conv3_top_dims);
  Tensor<float> * conv3_top_diff = Tensor<float>::CreateTensorGPU(conv3_top_dims);

  Pooling<float> pool3(2, MAX, 2);
  size_t pool3_top_dims[4];
  pool3.GetTopsDims({conv3_top_dims}, {pool3_top_dims});
  Tensor<float> * pool3_top = Tensor<float>::CreateTensorGPU(pool3_top_dims);
  Tensor<float> * pool3_top_diff = Tensor<float>::CreateTensorGPU(pool3_top_dims);

  Relu<float> relu3;
  size_t relu3_top_dims[4];
  relu3.GetTopsDims({pool3_top_dims}, {relu3_top_dims});
  Tensor<float> * relu3_top = Tensor<float>::CreateTensorGPU(relu3_top_dims);
  Tensor<float> * relu3_top_diff = Tensor<float>::CreateTensorGPU(relu3_top_dims);


  size_t to_fc4_dims[4];
  to_fc4_dims[0] = relu3_top_dims[0];
  to_fc4_dims[1] = 1;
  to_fc4_dims[2] = 1;
  to_fc4_dims[3] = relu3_top_dims[1]*relu3_top_dims[2]*relu3_top_dims[3];
  FC<float> fc4(to_fc4_dims[3],64);
  size_t fc4_top_dims[4];
  fc4.GetTopsDims({to_fc4_dims}, {fc4_top_dims});
  Tensor<float> * fc4_top = Tensor<float>::CreateTensorGPU(fc4_top_dims);
  Tensor<float> * fc4_top_diff = Tensor<float>::CreateTensorGPU(fc4_top_dims);

  FC<float> fc5(64, 10);
  size_t fc5_top_dims[4];
  fc5.GetTopsDims({fc4_top_dims}, {fc5_top_dims});
  Tensor<float> * fc5_top = Tensor<float>::CreateTensorGPU(fc5_top_dims);
  Tensor<float> * fc5_top_diff = Tensor<float>::CreateTensorGPU(fc5_top_dims);

  Softmax<float> softmax;
  size_t sm_top_dims[4];
  softmax.GetTopsDims({fc5_top_dims}, {sm_top_dims});
  Tensor<float> * sm_top = Tensor<float>::CreateTensorGPU(sm_top_dims);
  Tensor<float> * sm_top_diff = Tensor<float>::CreateTensorGPU(sm_top_dims);

  CrossEntropyLoss<float> cel_layer;
  size_t ce_top_dims[4];
  cel_layer.GetTopsDims({sm_top_dims, sm_top_dims}, {ce_top_dims});
  size_t ce_loss_dims[4] = {batch_size, 1, 1, 1};
  Tensor<float> * cel_top = Tensor<float>::CreateTensorGPU(ce_top_dims);
  Tensor<float> * cel_top_diff = Tensor<float>::CreateTensorCPU(ce_top_dims);
  Tensor<float>* cel_loss_diff = Tensor<float>::CreateTensorGPU(ce_loss_dims);


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


  printf("Forward inference .. \n");
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
  // flatten the tensor
  size_t relu3_top_dims_reshaped[4] = {relu3_top_dims[0], relu3_top_dims[3], relu3_top_dims[1], relu3_top_dims[2]};
  Tensor<float> * reshaped_relu3_top_cpu = Tensor<float>::CreateTensorCPU(relu3_top_dims_reshaped);
  Tensor<float> * relu3_top_cpu = Tensor<float>::TensorGPUtoCPU(relu3_top);
  for(int b = 0; b < relu3_top_dims_reshaped[0]; b++) {
    for(int c = 0; c < relu3_top_dims_reshaped[1]; c++) {
      for(int h = 0; h < relu3_top_dims_reshaped[2]; h++) {
        for(int w = 0; w < relu3_top_dims_reshaped[3]; w++) {
          reshaped_relu3_top_cpu->at(b, c, h, w) = relu3_top_cpu->at(b, h, w, c);
        }
      }
    }
  }
  Tensor<float> * reshaped_relu3_top = Tensor<float>::TensorCPUtoGPU(reshaped_relu3_top_cpu);
  Tensor<float> * reshaped_relu3_top_diff = Tensor<float>::CreateTensorGPU(to_fc4_dims);
  // flatten the tensor
  // Tensor<float>::ReshapeTensorGPU(relu3_top, to_fc4_dims);
  Tensor<float>::ReshapeTensorGPU(reshaped_relu3_top, to_fc4_dims);
  fc4.Forward({reshaped_relu3_top}, {fc4_top});
  // fc4.Forward({relu3_top}, {fc4_top});
  fc5.Forward({fc4_top}, {fc5_top});
  softmax.Forward({fc5_top}, {sm_top});
  cel_layer.Forward({sm_top, data_tops[1]}, {cel_top});


  for(int i = 0; i < 1; i++) {
    cel_layer.Backward({cel_top}, {cel_top}, {sm_top, data_tops[1]}, {sm_top_diff, cel_loss_diff});
    softmax.Backward({sm_top}, {sm_top_diff}, {fc5_top}, {fc5_top_diff});
    fc5.Backward({fc5_top}, {fc5_top_diff}, {fc4_top}, {fc4_top_diff});
    show_tensor<<<1, 1>>>(fc4_top_diff);
    fc4.Backward({fc4_top}, {fc4_top_diff}, {reshaped_relu3_top}, {reshaped_relu3_top_diff});
    // fc4.Backward({fc4_top}, {fc4_top_diff}, {relu3_top}, {relu3_top_diff});
    // Tensor<float>::ReshapeTensorGPU(relu3_top, relu3_top_dims);
    // Tensor<float>::ReshapeTensorGPU(relu3_top_diff, relu3_top_dims);
    // size_t reshaped_relu3_top_dims[4];
    // Tensor<float>::GetTensorGPUDims(reshaped_relu3_top, reshaped_relu3_top_dims);
    // printf("%d %d %d %d \n", reshaped_relu3_top_dims[0], reshaped_relu3_top_dims[1], reshaped_relu3_top_dims[2], reshaped_relu3_top_dims[3]);

    Tensor<float>::ReshapeTensorGPU(reshaped_relu3_top, relu3_top_dims_reshaped);
    Tensor<float>::ReshapeTensorGPU(reshaped_relu3_top_diff, relu3_top_dims_reshaped);

    Tensor<float> * relu3_top_diff_cpu = Tensor<float>::CreateTensorCPU(relu3_top_dims);
    
    // Tensor<float> * relu3_top_reshaped_cpu = Tensor<float>::TensorGPUtoCPU(reshaped_relu3_top);
    Tensor<float> * relu3_top_diff_reshaped_cpu = Tensor<float>::TensorGPUtoCPU(reshaped_relu3_top_diff);
    for(int b = 0; b < relu3_top_dims_reshaped[0]; b++) {
      for(int c = 0; c < relu3_top_dims_reshaped[1]; c++) {
        for(int h = 0; h < relu3_top_dims_reshaped[2]; h++) {
          for(int w = 0; w < relu3_top_dims_reshaped[3]; w++) {
            // relu3_top_cpu->at(b, h, w, c) = relu3_top_reshaped_cpu->at(b, c, h, w);
            relu3_top_diff_cpu->at(b, h, w, c) = relu3_top_diff_reshaped_cpu->at(b, c, h, w);
          }
        }
      }
    }

    Tensor<float> * relu3_top_diff = Tensor<float>::TensorCPUtoGPU(relu3_top_diff_cpu);

    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);
    // relu3.Backward({pool3_top}, {pool3_top_diff}, {reshaped_relu3_top}, {reshaped_relu3_top_diff});
    relu3.Backward({pool3_top}, {pool3_top_diff}, {relu3_top}, {relu3_top_diff});
    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);
    pool3.Backward({conv3_top}, {conv3_top_diff}, {pool3_top}, {pool3_top_diff});
    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);
    // conv3.Backward({relu2_top}, {relu2_top_diff}, {conv3_top}, {conv3_top_diff});
    conv3.Backward({conv3_top}, {conv3_top_diff}, {relu2_top}, {relu2_top_diff});
    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);
    // relu2.Backward({pool2_top}, {pool2_top_diff}, {relu2_top}, {relu2_top_diff});
    // cudaStatus = cudaGetLastError();
    // checkCudaErrors(cudaStatus);
    // pool2.Backward({conv2_top}, {conv2_top_diff}, {pool2_top}, {pool2_top_diff});
    // cudaStatus = cudaGetLastError();
    // checkCudaErrors(cudaStatus);
    // conv2.Backward({relu1_top}, {relu1_top_diff}, {conv2_top}, {conv2_top_diff});
    // cudaStatus = cudaGetLastError();
    // checkCudaErrors(cudaStatus);
    // relu1.Backward({pool1_top}, {pool1_top_diff}, {relu1_top}, {relu1_top_diff});
    // cudaStatus = cudaGetLastError();
    // checkCudaErrors(cudaStatus);
    // pool1.Backward({conv1_top}, {conv1_top_diff}, {pool1_top}, {pool1_top_diff});
    // cudaStatus = cudaGetLastError();
    // checkCudaErrors(cudaStatus);
    // conv1.Backward({data_tops[0]}, {data_top_diff0}, {conv1_top}, {conv1_top_diff});
    // cudaStatus = cudaGetLastError();
    // checkCudaErrors(cudaStatus);


    cudaStatus = cudaDeviceSynchronize();
    checkCudaErrors(cudaStatus);

  }




  printf("Prediction: \n");
  Tensor<float>* out = Tensor<float>::TensorGPUtoCPU(sm_top);
  for (int b = 0; b < out->GetDims()[0]; b++) {
    for (int h = 0; h < out->GetDims()[1]; h++) {
      for (int w = 0; w < out->GetDims()[2]; w++) {
        for (int c = 0; c < out->GetDims()[3]; c++) {
          if (c == 0) { printf("Airplane "); }
          else if (c == 1) { printf("Automobile "); } 
          else if (c == 2) { printf("Bird "); }
          else if (c == 3) { printf("Cat "); }
          else if (c == 4) { printf("Deer "); }
          else if (c == 5) { printf("Dog "); }
          else if (c == 6) { printf("Frog "); }
          else if (c == 7) { printf("Horse "); }
          else if (c == 8) { printf("Ship "); }
          else if (c == 9) { printf("truck "); }
          printf("probability: %1.4f \n", out->at(b,h,w,c));
        }
      }
    }
  }

}



int main() {
  demo_bp_cifar10_gpu();
}
