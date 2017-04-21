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


void test_lenet_gpu() {
  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);

  Session* session = Session::GetNewSession();
  session->gpu = true;

  size_t batch_size = 2;

  Data<float> data_layer(2, "tmp/test/img_list.txt");
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
}





int main() {
  test_lenet_gpu();
}
