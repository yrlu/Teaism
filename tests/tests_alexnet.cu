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
#include "layers/dropout.cu"
#include "layers/lrn.cu"
#include "layers/fc.cu"
#include "tmp/bitmap_image.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils/helper_cuda.h"
#include "utils/utils.cu"
#include "utils/load_model.hpp"

void test_alexnet_cpu() {
  printf("Start testing AlexNet with CPUs.\n");

  Session* session = Session::GetNewSession();
  session->gpu = false;
  session->batch_size = 128;

  size_t batch_size = session->batch_size;


  Data<float> data_layer(batch_size, "tmp/test/img_med_list.txt");
  size_t data_tops_dims0[4];
  size_t data_tops_dims1[4];
  data_layer.GetTopsDims({}, {data_tops_dims0, data_tops_dims1});
  std::vector<Tensor<float>*> data_tops;
  data_tops.push_back(Tensor<float>::CreateTensorCPU(data_tops_dims0));
  data_tops.push_back(Tensor<float>::CreateTensorCPU(data_tops_dims1));
  printf("data: (%d,%d,%d,%d)\n",data_tops_dims0[0],data_tops_dims0[1],data_tops_dims0[2],data_tops_dims0[3]);

  Conv2D<float> conv1(11,11,3,96,4, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv1_top_dims[4];
  conv1.GetTopsDims({data_tops_dims0}, {conv1_top_dims});

  Pooling<float> pool1(2,MAX,2);
  size_t pool1_top_dims[4];
  pool1.GetTopsDims({conv1_top_dims}, {pool1_top_dims});
  
  Relu<float> relu1;
  size_t relu1_top_dims[4];
  relu1.GetTopsDims({pool1_top_dims}, {relu1_top_dims});

  LRN<float> norm1;
  size_t norm1_top_dims[4];
  norm1.GetTopsDims({relu1_top_dims}, {norm1_top_dims});

  Conv2D<float> conv2(5,5,96,256,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv2_top_dims[4];
  conv2.GetTopsDims({norm1_top_dims}, {conv2_top_dims});

  Pooling<float> pool2(2, MAX, 2);
  size_t pool2_top_dims[4];
  pool2.GetTopsDims({conv2_top_dims}, {pool2_top_dims});

  Relu<float> relu2;
  size_t relu2_top_dims[4];
  relu2.GetTopsDims({pool2_top_dims}, {relu2_top_dims});

  LRN<float> norm2;
  size_t norm2_top_dims[4];
  norm2.GetTopsDims({relu2_top_dims}, {norm2_top_dims});

  Conv2D<float> conv3(3,3,256,384,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv3_top_dims[4];
  conv3.GetTopsDims({norm2_top_dims}, {conv3_top_dims});

  Relu<float> relu3;
  size_t relu3_top_dims[4];
  relu2.GetTopsDims({conv3_top_dims}, {relu3_top_dims});

  Conv2D<float> conv4(3,3,384,384,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv4_top_dims[4];
  conv4.GetTopsDims({relu3_top_dims}, {conv4_top_dims});

  Relu<float> relu4;
  size_t relu4_top_dims[4];
  relu4.GetTopsDims({conv4_top_dims}, {relu4_top_dims});

  Conv2D<float> conv5(3,3,384,256,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv5_top_dims[4];
  conv5.GetTopsDims({relu4_top_dims}, {conv5_top_dims});

  Pooling<float> pool5(2, MAX, 2);
  size_t pool5_top_dims[4];
  pool5.GetTopsDims({conv5_top_dims}, {pool5_top_dims});

  Relu<float> relu5;
  size_t relu5_top_dims[4];
  relu5.GetTopsDims({pool5_top_dims}, {relu5_top_dims});

  FC<float> fc6(relu5_top_dims[1]*relu5_top_dims[2]*relu5_top_dims[3],4096);
  size_t to_fc6_dims[4];
  to_fc6_dims[0] = relu5_top_dims[0];
  to_fc6_dims[1] = 1;
  to_fc6_dims[2] = 1;
  to_fc6_dims[3] = relu5_top_dims[1]*relu5_top_dims[2]*relu5_top_dims[3];
  size_t fc6_top_dims[4];
  fc6.GetTopsDims({to_fc6_dims}, {fc6_top_dims});

  Relu<float> relu6;
  size_t relu6_top_dims[4];
  relu6.GetTopsDims({fc6_top_dims}, {relu6_top_dims});

  Dropout<float> drop6;
  size_t drop6_top_dims[4];
  drop6.GetTopsDims({relu6_top_dims}, {drop6_top_dims});

  FC<float> fc7(4096,4096);
  size_t fc7_top_dims[4];
  fc7.GetTopsDims({drop6_top_dims}, {fc7_top_dims});

  Relu<float> relu7;
  size_t relu7_top_dims[4];
  relu7.GetTopsDims({fc7_top_dims}, {relu7_top_dims});

  Dropout<float> drop7;
  size_t drop7_top_dims[4];
  drop7.GetTopsDims({relu7_top_dims}, {drop7_top_dims});

  FC<float> fc8(4096,1000);
  size_t fc8_top_dims[4];
  fc8.GetTopsDims({drop7_top_dims}, {fc8_top_dims});

  Softmax<float> softmax;
  size_t sm_top_dims[4];
  softmax.GetTopsDims({fc8_top_dims}, {sm_top_dims});

  CrossEntropyLoss<float> cel;
  size_t cel_top_dims[4];
  cel.GetTopsDims({sm_top_dims, data_tops_dims1}, {cel_top_dims});

  printf("network finished setup: %3.1f ms \n", stopTimer());
 
 
  Tensor<float> * conv1_top = Tensor<float>::CreateTensorCPU(conv1_top_dims);
  printf("conv1: (%d,%d,%d,%d)\n",conv1_top_dims[0],conv1_top_dims[1],conv1_top_dims[2],conv1_top_dims[3]);

  Tensor<float> * pool1_top = Tensor<float>::CreateTensorCPU(pool1_top_dims);
  printf("pool1: (%d,%d,%d,%d)\n",pool1_top_dims[0],pool1_top_dims[1],pool1_top_dims[2],pool1_top_dims[3]);

  Tensor<float> * relu1_top = Tensor<float>::CreateTensorCPU(relu1_top_dims);
  printf("relu1: (%d,%d,%d,%d)\n",relu1_top_dims[0],relu1_top_dims[1],relu1_top_dims[2],relu1_top_dims[3]);

  Tensor<float> * norm1_top = Tensor<float>::CreateTensorCPU(norm1_top_dims);
  printf("norm1: (%d,%d,%d,%d)\n",norm1_top_dims[0],norm1_top_dims[1],norm1_top_dims[2],norm1_top_dims[3]);

  Tensor<float> * conv2_top = Tensor<float>::CreateTensorCPU(conv2_top_dims);
  printf("conv2: (%d,%d,%d,%d)\n",conv2_top_dims[0],conv2_top_dims[1],conv2_top_dims[2],conv2_top_dims[3]);

  Tensor<float> * pool2_top = Tensor<float>::CreateTensorCPU(pool2_top_dims);
  printf("pool2: (%d,%d,%d,%d)\n",pool2_top_dims[0],pool2_top_dims[1],pool2_top_dims[2],pool2_top_dims[3]);

  Tensor<float> * relu2_top = Tensor<float>::CreateTensorCPU(relu2_top_dims);
  printf("relu2: (%d,%d,%d,%d)\n",relu2_top_dims[0],relu2_top_dims[1],relu2_top_dims[2],relu2_top_dims[3]);

  Tensor<float> * norm2_top = Tensor<float>::CreateTensorCPU(norm2_top_dims);
  printf("norm2: (%d,%d,%d,%d)\n",norm2_top_dims[0],norm2_top_dims[1],norm2_top_dims[2],norm2_top_dims[3]);

  Tensor<float> * conv3_top = Tensor<float>::CreateTensorCPU(conv3_top_dims);
  printf("conv3: (%d,%d,%d,%d)\n",conv3_top_dims[0],conv3_top_dims[1],conv3_top_dims[2],conv3_top_dims[3]);

  Tensor<float> * relu3_top = Tensor<float>::CreateTensorCPU(relu3_top_dims);
  printf("relu3: (%d,%d,%d,%d)\n",relu3_top_dims[0],relu3_top_dims[1],relu3_top_dims[2],relu3_top_dims[3]);

  Tensor<float> * conv4_top = Tensor<float>::CreateTensorCPU(conv4_top_dims);
  printf("conv4: (%d,%d,%d,%d)\n",conv4_top_dims[0],conv4_top_dims[1],conv4_top_dims[2],conv4_top_dims[3]);

  Tensor<float> * relu4_top = Tensor<float>::CreateTensorCPU(relu4_top_dims);
  printf("relu4: (%d,%d,%d,%d)\n",relu4_top_dims[0],relu4_top_dims[1],relu4_top_dims[2],relu4_top_dims[3]);

  Tensor<float> * conv5_top = Tensor<float>::CreateTensorCPU(conv5_top_dims);
  printf("conv5: (%d,%d,%d,%d)\n",conv5_top_dims[0],conv5_top_dims[1],conv5_top_dims[2],conv5_top_dims[3]);

  Tensor<float> * pool5_top = Tensor<float>::CreateTensorCPU(pool5_top_dims);
  printf("pool5: (%d,%d,%d,%d)\n",pool5_top_dims[0],pool5_top_dims[1],pool5_top_dims[2],pool5_top_dims[3]);

  Tensor<float> * relu5_top = Tensor<float>::CreateTensorCPU(relu5_top_dims);
  printf("relu5: (%d,%d,%d,%d)\n",relu5_top_dims[0],relu5_top_dims[1],relu5_top_dims[2],relu5_top_dims[3]);

  Tensor<float> * fc6_top = Tensor<float>::CreateTensorCPU(fc6_top_dims);
  Tensor<float> * relu6_top = Tensor<float>::CreateTensorCPU(relu6_top_dims);
  Tensor<float> * drop6_top = Tensor<float>::CreateTensorCPU(drop6_top_dims);
  Tensor<float> * fc7_top = Tensor<float>::CreateTensorCPU(fc7_top_dims);
  Tensor<float> * relu7_top = Tensor<float>::CreateTensorCPU(relu7_top_dims);
  Tensor<float> * drop7_top = Tensor<float>::CreateTensorCPU(drop7_top_dims);
  Tensor<float> * fc8_top = Tensor<float>::CreateTensorCPU(fc8_top_dims);
  Tensor<float> * sm_top = Tensor<float>::CreateTensorCPU(sm_top_dims);
  Tensor<float> * cel_top = Tensor<float>::CreateTensorCPU(cel_top_dims);




  startTimer();
  data_layer.Forward(std::vector<Tensor<float>*> (), data_tops);
  printf("data forward: %3.1f ms \n", stopTimer()); startTimer();
  conv1.Forward({data_tops[0]}, {conv1_top});
  printf("conv1 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool1.Forward({conv1_top}, {pool1_top});
  printf("pool1 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu1.Forward({pool1_top}, {relu1_top});
  printf("relu1 forward: %3.1f ms \n", stopTimer()); startTimer();
  norm1.Forward({relu1_top}, {norm1_top});
  printf("norm1 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv2.Forward({relu1_top}, {conv2_top});
  printf("conv2 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool2.Forward({conv2_top}, {pool2_top});
  printf("pool2 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu2.Forward({pool2_top}, {relu2_top});
  printf("relu2 forward: %3.1f ms \n", stopTimer()); startTimer();
  norm2.Forward({relu2_top}, {norm2_top});
  printf("norm2 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv3.Forward({relu2_top}, {conv3_top});
  printf("conv3 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu3.Forward({conv3_top}, {relu3_top});
  printf("relu3 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv4.Forward({relu3_top}, {conv4_top});
  printf("conv4 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu4.Forward({conv4_top}, {relu4_top});
  printf("relu4 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv5.Forward({relu4_top}, {conv5_top});
  printf("conv5 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool5.Forward({conv5_top}, {pool5_top});
  printf("pool5 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu5.Forward({pool5_top}, {relu5_top});
  printf("relu5 forward: %3.1f ms \n", stopTimer()); startTimer();
  fc6.Forward({relu5_top}, {fc6_top});
  printf("fc6 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu6.Forward({fc6_top}, {relu6_top});
  printf("relu6 forward: %3.1f ms \n", stopTimer()); startTimer();
  drop6.Forward({relu6_top}, {drop6_top});
  printf("drop6 forward: %3.1f ms \n", stopTimer()); startTimer();
  fc7.Forward({drop6_top}, {fc7_top});
  printf("fc7 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu7.Forward({fc7_top}, {relu7_top});
  printf("relu7 forward: %3.1f ms \n", stopTimer()); startTimer();
  drop7.Forward({relu7_top}, {drop7_top});
  printf("drop7 forward: %3.1f ms \n", stopTimer()); startTimer();
  fc8.Forward({drop7_top}, {fc8_top});
  printf("fc8 forward: %3.1f ms \n", stopTimer()); startTimer();
  softmax.Forward({fc8_top}, {sm_top});
  printf("softmax forward: %3.1f ms \n", stopTimer()); startTimer();
  cel.Forward({sm_top, data_tops[1]}, {cel_top});
  printf("cel forward: %3.1f ms \n", stopTimer());


  startTimer();
  data_layer.Forward(std::vector<Tensor<float>*> (), data_tops);
  conv1.Forward({data_tops[0]}, {conv1_top});
  pool1.Forward({conv1_top}, {pool1_top});
  relu1.Forward({pool1_top}, {relu1_top});
  norm1.Forward({relu1_top}, {norm1_top});
  conv2.Forward({relu1_top}, {conv2_top});
  pool2.Forward({conv2_top}, {pool2_top});
  relu2.Forward({pool2_top}, {relu2_top});
  norm2.Forward({relu2_top}, {norm2_top});
  conv3.Forward({relu2_top}, {conv3_top});
  relu3.Forward({conv3_top}, {relu3_top});
  conv4.Forward({relu3_top}, {conv4_top});
  relu4.Forward({conv4_top}, {relu4_top});
  conv5.Forward({relu4_top}, {conv5_top});
  pool5.Forward({conv5_top}, {pool5_top});
  relu5.Forward({pool5_top}, {relu5_top});
  fc6.Forward({relu5_top}, {fc6_top});
  relu6.Forward({fc6_top}, {relu6_top});
  drop6.Forward({relu6_top}, {drop6_top});
  fc7.Forward({drop6_top}, {fc7_top});
  relu7.Forward({fc7_top}, {relu7_top});
  drop7.Forward({relu7_top}, {drop7_top});
  fc8.Forward({drop7_top}, {fc8_top});
  softmax.Forward({fc8_top}, {sm_top});
  cel.Forward({sm_top, data_tops[1]}, {cel_top});
  printf("finished forward: %3.1f ms \n", stopTimer());


}



void test_alexnet_gpu() {
  printf("Start testing AlexNet with GPUs.\n");

  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);
  show_mem(cudaStatus);
  startTimer();

  Session* session = Session::GetNewSession();
  session->gpu = true;
  session->batch_size = 128;

  size_t batch_size = session->batch_size;


  Data<float> data_layer(batch_size, "tmp/test/img_med_list.txt");
  size_t data_tops_dims0[4];
  size_t data_tops_dims1[4];
  data_layer.GetTopsDims({}, {data_tops_dims0, data_tops_dims1});
  std::vector<Tensor<float>*> data_tops;
  data_tops.push_back(Tensor<float>::CreateTensorGPU(data_tops_dims0));
  data_tops.push_back(Tensor<float>::CreateTensorGPU(data_tops_dims1));
  printf("data: (%d,%d,%d,%d)\n",data_tops_dims0[0],data_tops_dims0[1],data_tops_dims0[2],data_tops_dims0[3]);

  Conv2D<float> conv1(11,11,3,96,4, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv1_top_dims[4];
  conv1.GetTopsDims({data_tops_dims0}, {conv1_top_dims});

  Pooling<float> pool1(2,MAX,2);
  size_t pool1_top_dims[4];
  pool1.GetTopsDims({conv1_top_dims}, {pool1_top_dims});
  
  Relu<float> relu1;
  size_t relu1_top_dims[4];
  relu1.GetTopsDims({pool1_top_dims}, {relu1_top_dims});

  LRN<float> norm1;
  size_t norm1_top_dims[4];
  norm1.GetTopsDims({relu1_top_dims}, {norm1_top_dims});

  Conv2D<float> conv2(5,5,96,256,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv2_top_dims[4];
  conv2.GetTopsDims({norm1_top_dims}, {conv2_top_dims});

  Pooling<float> pool2(2, MAX, 2);
  size_t pool2_top_dims[4];
  pool2.GetTopsDims({conv2_top_dims}, {pool2_top_dims});

  Relu<float> relu2;
  size_t relu2_top_dims[4];
  relu2.GetTopsDims({pool2_top_dims}, {relu2_top_dims});

  LRN<float> norm2;
  size_t norm2_top_dims[4];
  norm2.GetTopsDims({relu2_top_dims}, {norm2_top_dims});

  Conv2D<float> conv3(3,3,256,384,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv3_top_dims[4];
  conv3.GetTopsDims({norm2_top_dims}, {conv3_top_dims});

  Relu<float> relu3;
  size_t relu3_top_dims[4];
  relu2.GetTopsDims({conv3_top_dims}, {relu3_top_dims});

  Conv2D<float> conv4(3,3,384,384,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv4_top_dims[4];
  conv4.GetTopsDims({relu3_top_dims}, {conv4_top_dims});

  Relu<float> relu4;
  size_t relu4_top_dims[4];
  relu4.GetTopsDims({conv4_top_dims}, {relu4_top_dims});

  Conv2D<float> conv5(3,3,384,256,1, new GaussianKernelInitializer<float>(0.1), SAME);
  size_t conv5_top_dims[4];
  conv5.GetTopsDims({relu4_top_dims}, {conv5_top_dims});

  Pooling<float> pool5(2, MAX, 2);
  size_t pool5_top_dims[4];
  pool5.GetTopsDims({conv5_top_dims}, {pool5_top_dims});

  Relu<float> relu5;
  size_t relu5_top_dims[4];
  relu5.GetTopsDims({pool5_top_dims}, {relu5_top_dims});

  FC<float> fc6(relu5_top_dims[1]*relu5_top_dims[2]*relu5_top_dims[3],4096);
  size_t to_fc6_dims[4];
  to_fc6_dims[0] = relu5_top_dims[0];
  to_fc6_dims[1] = 1;
  to_fc6_dims[2] = 1;
  to_fc6_dims[3] = relu5_top_dims[1]*relu5_top_dims[2]*relu5_top_dims[3];
  size_t fc6_top_dims[4];
  fc6.GetTopsDims({to_fc6_dims}, {fc6_top_dims});

  Relu<float> relu6;
  size_t relu6_top_dims[4];
  relu6.GetTopsDims({fc6_top_dims}, {relu6_top_dims});

  Dropout<float> drop6;
  size_t drop6_top_dims[4];
  drop6.GetTopsDims({relu6_top_dims}, {drop6_top_dims});

  FC<float> fc7(4096,4096);
  size_t fc7_top_dims[4];
  fc7.GetTopsDims({drop6_top_dims}, {fc7_top_dims});

  Relu<float> relu7;
  size_t relu7_top_dims[4];
  relu7.GetTopsDims({fc7_top_dims}, {relu7_top_dims});

  Dropout<float> drop7;
  size_t drop7_top_dims[4];
  drop7.GetTopsDims({relu7_top_dims}, {drop7_top_dims});

  FC<float> fc8(4096,1000);
  size_t fc8_top_dims[4];
  fc8.GetTopsDims({drop7_top_dims}, {fc8_top_dims});

  Softmax<float> softmax;
  size_t sm_top_dims[4];
  softmax.GetTopsDims({fc8_top_dims}, {sm_top_dims});

  CrossEntropyLoss<float> cel;
  size_t cel_top_dims[4];
  cel.GetTopsDims({sm_top_dims, data_tops_dims1}, {cel_top_dims});

  printf("network finished setup: %3.1f ms \n", stopTimer());
  show_mem(cudaStatus);
  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);
 
 
  Tensor<float> * conv1_top = Tensor<float>::CreateTensorGPU(conv1_top_dims);
  printf("conv1: (%d,%d,%d,%d)\n",conv1_top_dims[0],conv1_top_dims[1],conv1_top_dims[2],conv1_top_dims[3]);

  Tensor<float> * pool1_top = Tensor<float>::CreateTensorGPU(pool1_top_dims);
  printf("pool1: (%d,%d,%d,%d)\n",pool1_top_dims[0],pool1_top_dims[1],pool1_top_dims[2],pool1_top_dims[3]);

  Tensor<float> * relu1_top = Tensor<float>::CreateTensorGPU(relu1_top_dims);
  printf("relu1: (%d,%d,%d,%d)\n",relu1_top_dims[0],relu1_top_dims[1],relu1_top_dims[2],relu1_top_dims[3]);

  Tensor<float> * norm1_top = Tensor<float>::CreateTensorGPU(norm1_top_dims);
  printf("norm1: (%d,%d,%d,%d)\n",norm1_top_dims[0],norm1_top_dims[1],norm1_top_dims[2],norm1_top_dims[3]);

  Tensor<float> * conv2_top = Tensor<float>::CreateTensorGPU(conv2_top_dims);
  printf("conv2: (%d,%d,%d,%d)\n",conv2_top_dims[0],conv2_top_dims[1],conv2_top_dims[2],conv2_top_dims[3]);

  Tensor<float> * pool2_top = Tensor<float>::CreateTensorGPU(pool2_top_dims);
  printf("pool2: (%d,%d,%d,%d)\n",pool2_top_dims[0],pool2_top_dims[1],pool2_top_dims[2],pool2_top_dims[3]);

  Tensor<float> * relu2_top = Tensor<float>::CreateTensorGPU(relu2_top_dims);
  printf("relu2: (%d,%d,%d,%d)\n",relu2_top_dims[0],relu2_top_dims[1],relu2_top_dims[2],relu2_top_dims[3]);

  Tensor<float> * norm2_top = Tensor<float>::CreateTensorGPU(norm2_top_dims);
  printf("norm2: (%d,%d,%d,%d)\n",norm2_top_dims[0],norm2_top_dims[1],norm2_top_dims[2],norm2_top_dims[3]);

  Tensor<float> * conv3_top = Tensor<float>::CreateTensorGPU(conv3_top_dims);
  printf("conv3: (%d,%d,%d,%d)\n",conv3_top_dims[0],conv3_top_dims[1],conv3_top_dims[2],conv3_top_dims[3]);

  Tensor<float> * relu3_top = Tensor<float>::CreateTensorGPU(relu3_top_dims);
  printf("relu3: (%d,%d,%d,%d)\n",relu3_top_dims[0],relu3_top_dims[1],relu3_top_dims[2],relu3_top_dims[3]);

  Tensor<float> * conv4_top = Tensor<float>::CreateTensorGPU(conv4_top_dims);
  printf("conv4: (%d,%d,%d,%d)\n",conv4_top_dims[0],conv4_top_dims[1],conv4_top_dims[2],conv4_top_dims[3]);

  Tensor<float> * relu4_top = Tensor<float>::CreateTensorGPU(relu4_top_dims);
  printf("relu4: (%d,%d,%d,%d)\n",relu4_top_dims[0],relu4_top_dims[1],relu4_top_dims[2],relu4_top_dims[3]);

  Tensor<float> * conv5_top = Tensor<float>::CreateTensorGPU(conv5_top_dims);
  printf("conv5: (%d,%d,%d,%d)\n",conv5_top_dims[0],conv5_top_dims[1],conv5_top_dims[2],conv5_top_dims[3]);

  Tensor<float> * pool5_top = Tensor<float>::CreateTensorGPU(pool5_top_dims);
  printf("pool5: (%d,%d,%d,%d)\n",pool5_top_dims[0],pool5_top_dims[1],pool5_top_dims[2],pool5_top_dims[3]);

  Tensor<float> * relu5_top = Tensor<float>::CreateTensorGPU(relu5_top_dims);
  printf("relu5: (%d,%d,%d,%d)\n",relu5_top_dims[0],relu5_top_dims[1],relu5_top_dims[2],relu5_top_dims[3]);

  Tensor<float> * fc6_top = Tensor<float>::CreateTensorGPU(fc6_top_dims);
  Tensor<float> * relu6_top = Tensor<float>::CreateTensorGPU(relu6_top_dims);
  Tensor<float> * drop6_top = Tensor<float>::CreateTensorGPU(drop6_top_dims);
  Tensor<float> * fc7_top = Tensor<float>::CreateTensorGPU(fc7_top_dims);
  Tensor<float> * relu7_top = Tensor<float>::CreateTensorGPU(relu7_top_dims);
  Tensor<float> * drop7_top = Tensor<float>::CreateTensorGPU(drop7_top_dims);
  Tensor<float> * fc8_top = Tensor<float>::CreateTensorGPU(fc8_top_dims);
  Tensor<float> * sm_top = Tensor<float>::CreateTensorGPU(sm_top_dims);
  Tensor<float> * cel_top = Tensor<float>::CreateTensorGPU(cel_top_dims);




  startTimer();
  data_layer.Forward(std::vector<Tensor<float>*> (), data_tops);
  printf("data forward: %3.1f ms \n", stopTimer()); startTimer();
  conv1.Forward({data_tops[0]}, {conv1_top});
  printf("conv1 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool1.Forward({conv1_top}, {pool1_top});
  printf("pool1 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu1.Forward({pool1_top}, {relu1_top});
  printf("relu1 forward: %3.1f ms \n", stopTimer()); startTimer();
  norm1.Forward({relu1_top}, {norm1_top});
  printf("norm1 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv2.Forward({relu1_top}, {conv2_top});
  printf("conv2 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool2.Forward({conv2_top}, {pool2_top});
  printf("pool2 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu2.Forward({pool2_top}, {relu2_top});
  printf("relu2 forward: %3.1f ms \n", stopTimer()); startTimer();
  norm2.Forward({relu2_top}, {norm2_top});
  printf("norm2 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv3.Forward({relu2_top}, {conv3_top});
  printf("conv3 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu3.Forward({conv3_top}, {relu3_top});
  printf("relu3 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv4.Forward({relu3_top}, {conv4_top});
  printf("conv4 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu4.Forward({conv4_top}, {relu4_top});
  printf("relu4 forward: %3.1f ms \n", stopTimer()); startTimer();
  conv5.Forward({relu4_top}, {conv5_top});
  printf("conv5 forward: %3.1f ms \n", stopTimer()); startTimer();
  pool5.Forward({conv5_top}, {pool5_top});
  printf("pool5 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu5.Forward({pool5_top}, {relu5_top});
  printf("relu5 forward: %3.1f ms \n", stopTimer()); startTimer();
  fc6.Forward({relu5_top}, {fc6_top});
  printf("fc6 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu6.Forward({fc6_top}, {relu6_top});
  printf("relu6 forward: %3.1f ms \n", stopTimer()); startTimer();
  drop6.Forward({relu6_top}, {drop6_top});
  printf("drop6 forward: %3.1f ms \n", stopTimer()); startTimer();
  fc7.Forward({drop6_top}, {fc7_top});
  printf("fc7 forward: %3.1f ms \n", stopTimer()); startTimer();
  relu7.Forward({fc7_top}, {relu7_top});
  printf("relu7 forward: %3.1f ms \n", stopTimer()); startTimer();
  drop7.Forward({relu7_top}, {drop7_top});
  printf("drop7 forward: %3.1f ms \n", stopTimer()); startTimer();
  fc8.Forward({drop7_top}, {fc8_top});
  printf("fc8 forward: %3.1f ms \n", stopTimer()); startTimer();
  softmax.Forward({fc8_top}, {sm_top});
  printf("softmax forward: %3.1f ms \n", stopTimer()); startTimer();
  cel.Forward({sm_top, data_tops[1]}, {cel_top});
  printf("cel forward: %3.1f ms \n", stopTimer());
  show_mem(cudaStatus);


  data_layer.Forward(std::vector<Tensor<float>*> (), data_tops);
  startTimer();
  conv1.Forward({data_tops[0]}, {conv1_top});
  pool1.Forward({conv1_top}, {pool1_top});
  relu1.Forward({pool1_top}, {relu1_top});
  norm1.Forward({relu1_top}, {norm1_top});
  conv2.Forward({relu1_top}, {conv2_top});
  pool2.Forward({conv2_top}, {pool2_top});
  relu2.Forward({pool2_top}, {relu2_top});
  norm2.Forward({relu2_top}, {norm2_top});
  conv3.Forward({relu2_top}, {conv3_top});
  relu3.Forward({conv3_top}, {relu3_top});
  conv4.Forward({relu3_top}, {conv4_top});
  relu4.Forward({conv4_top}, {relu4_top});
  conv5.Forward({relu4_top}, {conv5_top});
  pool5.Forward({conv5_top}, {pool5_top});
  relu5.Forward({pool5_top}, {relu5_top});
  fc6.Forward({relu5_top}, {fc6_top});
  relu6.Forward({fc6_top}, {relu6_top});
  drop6.Forward({relu6_top}, {drop6_top});
  fc7.Forward({drop6_top}, {fc7_top});
  relu7.Forward({fc7_top}, {relu7_top});
  drop7.Forward({relu7_top}, {drop7_top});
  fc8.Forward({drop7_top}, {fc8_top});
  softmax.Forward({fc8_top}, {sm_top});
  cel.Forward({sm_top, data_tops[1]}, {cel_top});
  printf("finished forward: %3.1f ms \n", stopTimer());
  show_mem(cudaStatus);

  cudaStatus = cudaGetLastError();
  checkCudaErrors(cudaStatus);

  show_mem(cudaStatus);
}





int main() {
//  test_alexnet_cpu();
  test_alexnet_gpu();
}
