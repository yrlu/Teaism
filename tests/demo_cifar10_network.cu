#include <stdio.h>
#include <assert.h>
#include "basics/tensor.cu"
#include "basics/session.hpp"
#include "basics/network.cu"
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



__global__ void show_tensor(Tensor<double> * tensor) {
  size_t d1 = tensor->GetDims()[0];
  size_t d2 = tensor->GetDims()[1];
  size_t d3 = tensor->GetDims()[2];
  size_t d4 = tensor->GetDims()[3];

        for(int k = 0; k < d3; k++) {
    for(int l = 0; l < d4; l++) {
      for(int j = 0; j < d2; j++) {
  for(int i = 0; i < d1; i++) {
          printf("%e ", tensor->at(i, j, k, l));
        }
        printf("\n");
      }
      printf("\n");
    }
    printf("\n");
  }
}




void print_acc(int iter, int batch_size, Tensor<double>* sm_top_gpu, Tensor<double> * labels_gpu) {
  Tensor<double>* sm_top_cpu = Tensor<double>::TensorGPUtoCPU(sm_top_gpu);
  Tensor<double>* label_cpu = Tensor<double>::TensorGPUtoCPU(labels_gpu);

  // batch, 1, 1, 1
  // batch, 1, 1, 1
  double cnt = 0;
  for(int i = 0; i < batch_size; i++) {
    double max_val = sm_top_cpu->at(i, 0, 0, 0);
    int label = 0;
    for(int j = 0; j < 10; j++) {
      // printf("%f ", sm_top_cpu->at(i, 0, 0, j));
      if(sm_top_cpu->at(i, 0, 0, j) > max_val) {
        max_val = sm_top_cpu->at(i, 0, 0, j);
        label = j;
      }
    }

    // printf("predicted label: %d, ground truth label: %d \n", label, (int)label_cpu->at(i, 0, 0, 0));
    if(label == (int)label_cpu->at(i, 0, 0, 0)) {
      cnt += 1;
    }
  }

  double acc = cnt / (double) batch_size;
  printf("iteration %d accuracy: %d/%d %f \n", iter, (int)cnt, batch_size, acc);

  delete sm_top_cpu;
  delete label_cpu;
}



void demo_bp_cifar10_gpu() {
  printf("Start training convolutional networks on cifar10\n\n");

  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);
  startTimer();

  Session* session = Session::GetNewSession();
  session->gpu = true;
  session->batch_size = 1;
  session->lr = 0.0002;
  size_t batch_size = session->batch_size;

  Data<double>* data_layer = new Data<double>(batch_size, "datasets/cifar10/train.txt");
  Conv2D<double>* conv1 = new Conv2D<double>
    (5,5,3,32,1, new GaussianKernelInitializer<double>(0.0001), SAME);
  Pooling<double>* pool1 = new Pooling<double>(2, MAX, 2);
  Relu<double>* relu1 = new Relu<double>;
  Conv2D<double>* conv2 = new Conv2D<double>
    (5,5,32,32,1, new GaussianKernelInitializer<double>(0.01), SAME);
  Pooling<double>* pool2 = new Pooling<double>(2, MAX, 2);
  Relu<double>* relu2 = new Relu<double>;
  Conv2D<double>* conv3 = new Conv2D<double>
    (5,5,32,64,1, new GaussianKernelInitializer<double>(0.01), SAME);
  Pooling<double>* pool3 = new Pooling<double>(2, MAX, 2);
  Relu<double>* relu3 = new Relu<double>;
  FC<double>* fc4 = new FC<double>
    (4*4*64,64, new GaussianKernelInitializer<double>(0.1));
  FC<double>* fc5 = new FC<double>
    (64, 10, new GaussianKernelInitializer<double>(0.1));
  Softmax<double>* softmax = new Softmax<double>;
  CrossEntropyLoss<double>* cel_layer = new CrossEntropyLoss<double>;

  Network<double> network({data_layer, conv1, pool1, relu1, 
                           conv2, pool2, relu2,
                           conv3, pool3, relu3,
                           fc4, fc5, softmax, cel_layer});

  printf("network finished setup: %3.1f ms \n", stopTimer());
  show_mem(cudaStatus);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);


  for(int iter = 0; iter < 10000; iter++) {
    startTimer();
    network.Forward();
    network.Backward();
    printf("iteration: %d\n", iter);
    Tensor<double>* cel_top = network.GetLayerData(13)->tops[0];
    if (iter % 1 == 0) {
      show_tensor<<<1,1>>>(cel_top);
    }

    Tensor<double>* sm_top = network.GetLayerData(12)->tops[0];
    Tensor<double>* labels = network.GetLayerData(0)->tops[1];
    print_acc(iter, batch_size, sm_top, labels);
    printf("iteration time: %3.1f ms \n", stopTimer());
  }

  printf("Prediction: \n");
  Tensor<double>* sm_top = network.GetLayerData(12)->tops[0];
  Tensor<double>* out = Tensor<double>::TensorGPUtoCPU(sm_top);
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
