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
#include "tmp/bitmap_image.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils/helper_cuda.h"

// show memory usage of GPU
void show_mem(cudaError_t cuda_status) {
  size_t free_byte;
  size_t total_byte;
  cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
  if(cudaSuccess!=cuda_status) {
  	printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
  	return;
  }
  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;

  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
  used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}


// used by startTimer() and stopTimer()
cudaEvent_t start, stop;

void startTimer() {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

/** Return elapsed time (in ms) since startTime() was called */
float stopTimer() {
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	return time;
}

void test_lenet_gpu() {
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);
  show_mem(cudaStatus);
  startTimer();

  Session* session = Session::GetNewSession();
  session->gpu = true;

  size_t batch_size = 64;


  Data<float> data_layer(batch_size, "tmp/test/img_list.txt");
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

  Pooling<float> pool1(2);
  size_t pool1_top_dims[4];
  pool1.GetTopsDims({conv1_top_dims}, {pool1_top_dims});
  Tensor<float> * pool1_top = Tensor<float>::CreateTensorGPU(pool1_top_dims);
  
  Relu<float> relu1;
  size_t relu1_top_dims[4];
  relu1.GetTopsDims({pool1_top_dims}, {relu1_top_dims});
  Tensor<float> * relu1_top = Tensor<float>::CreateTensorGPU(relu1_top_dims);

  Conv2D<float> conv2(5,5,32,64,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv2_top_dims[4];
  conv2.GetTopsDims({relu1_top_dims}, {conv2_top_dims});
  Tensor<float> * conv2_top = Tensor<float>::CreateTensorGPU(conv2_top_dims);

  Pooling<float> pool2(2);
  size_t pool2_top_dims[4];
  pool2.GetTopsDims({conv2_top_dims}, {pool2_top_dims});
  Tensor<float> * pool2_top = Tensor<float>::CreateTensorGPU(pool2_top_dims);

  Relu<float> relu2;
  size_t relu2_top_dims[4];
  relu2.GetTopsDims({pool2_top_dims}, {relu2_top_dims});
  Tensor<float> * relu2_top = Tensor<float>::CreateTensorGPU(relu2_top_dims);

  Conv2D<float> fc3(7,7,64,1024,1, new GaussianKernelInitializer<float>(0.1), VALID);
  size_t fc3_top_dims[4];
  fc3.GetTopsDims({relu2_top_dims}, {fc3_top_dims});
  Tensor<float> * fc3_top = Tensor<float>::CreateTensorGPU(fc3_top_dims);

  Relu<float> relu3;
  size_t relu3_top_dims[4];
  relu3.GetTopsDims({fc3_top_dims}, {relu3_top_dims});
  Tensor<float> * relu3_top = Tensor<float>::CreateTensorGPU(relu3_top_dims);

  Conv2D<float> fc4(1,1,1024,10,1, new GaussianKernelInitializer<float>(0.1), VALID);
  size_t fc4_top_dims[4];
  fc4.GetTopsDims({relu3_top_dims}, {fc4_top_dims});
  Tensor<float> * fc4_top = Tensor<float>::CreateTensorGPU(fc4_top_dims);

  Softmax<float> softmax;
  size_t sm_top_dims[4];
  softmax.GetTopsDims({fc4_top_dims}, {sm_top_dims});
  Tensor<float> * sm_top = Tensor<float>::CreateTensorGPU(sm_top_dims);

  CrossEntropyLoss<float> cel;
  size_t cel_top_dims[4];
  cel.GetTopsDims({sm_top_dims, data_tops_dims1}, {cel_top_dims});
  Tensor<float> * cel_top = Tensor<float>::CreateTensorGPU(cel_top_dims);

  printf("network finished setup: %3.1f ms \n", stopTimer());
  show_mem(cudaStatus);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
  

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
  fc3.Forward({relu2_top}, {fc3_top});
  printf("fc3 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu3.Forward({fc3_top}, {relu3_top});
  printf("relu3 forward: %3.1f ms \n", stopTimer()); startTimer();
  fc4.Forward({relu3_top}, {fc4_top});
  printf("fc4 forward: %3.1f ms \n", stopTimer()); startTimer();
  softmax.Forward({fc4_top}, {sm_top});
  printf("softmax forward: %3.1f ms \n", stopTimer()); startTimer();
  cel.Forward({sm_top, data_tops[1]}, {cel_top});
  printf("cel forward: %3.1f ms \n", stopTimer());
  show_mem(cudaStatus);


  startTimer();
  data_layer.Forward(std::vector<Tensor<float>*> (), data_tops);
  conv1.Forward({data_tops[0]}, {conv1_top});
  pool1.Forward({conv1_top}, {pool1_top});
  relu1.Forward({pool1_top}, {relu1_top});
  conv2.Forward({relu1_top}, {conv2_top});
  pool2.Forward({conv2_top}, {pool2_top});
  relu2.Forward({pool2_top}, {relu2_top});
  fc3.Forward({relu2_top}, {fc3_top});
  relu3.Forward({fc3_top}, {relu3_top});
  fc4.Forward({relu3_top}, {fc4_top});
  softmax.Forward({fc4_top}, {sm_top});
  cel.Forward({sm_top, data_tops[1]}, {cel_top});
  printf("finished forward: %3.1f ms \n", stopTimer());
  show_mem(cudaStatus);



  Tensor<float> * output_cpu = Tensor<float>::TensorGPUtoCPU(sm_top);
  Tensor<float> * fc4_cpu = Tensor<float>::TensorGPUtoCPU(fc4_top);


  // printf("%f \n", sm_top->at(0,0,0,0));
  for(int i = 0; i < sm_top_dims[0]; i++) {
  	for(int j = 0; j < sm_top_dims[3]; j++) {
  	  printf("%f ", output_cpu->at(i, 0, 0, j));
  	}
  	printf("\n");
  }

  for(int i = 0; i < fc4_top_dims[0]; i++) {
  	for(int j = 0; j < fc4_top_dims[3]; j++) {
  	  printf("%f ", fc4_cpu->at(i, 0, 0, j));
  	}
  	printf("\n");
  }




  printf("%d %d %d %d \n", fc4_top_dims[0], fc4_top_dims[1], fc4_top_dims[2], fc4_top_dims[3]);
  printf("%d %d %d %d \n", data_tops_dims1[0], data_tops_dims1[1], data_tops_dims1[2], data_tops_dims1[3]);
  printf("%d %d %d %d \n", cel_top_dims[0], cel_top_dims[1], cel_top_dims[2], cel_top_dims[3]);
  printf("%d %d %d %d \n", sm_top_dims[0], sm_top_dims[1], sm_top_dims[2], sm_top_dims[3]);
  

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  show_mem(cudaStatus);
}





int main() {
  test_lenet_gpu();
}
