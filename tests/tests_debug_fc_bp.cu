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
#include "initializers/const_initializer.cu"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils/helper_cuda.h"
#include "utils/utils.cu"


void show_fc_tensor_cpu(Tensor<float> * fc_tensor_cpu) {
  for(int b = 0; b < fc_tensor_cpu->GetDims()[0]) {
    for(int i = 0; i < fc_tensor_cpu->GetDims()[3]) {
      printf("%f ", fc_tensor_cpu->at(b, 0, 0, i));
    }
    printf("\n");
  }
  printf("\n");
}


void show_fc_w_cpu(Tensor<float> * fc_w_cpu) {
  for(int i = 0; i < fc_w_cpu->GetDims()[3]) {
    for(int o = 0; o < fc_w_cpu->GetDims()[2]) {
      printf("%f ", fc_w_cpu->at(0,0,o,i));
    }
    printf("\n");
  }
  printf("\n");
}

void show_fc_b_cpu(Tensor<float> * fc_b_cpu) {
  for(int o = 0; o < fc_b_cpu->GetDims()[3]) {
    printf("%f ", fc_b_cpu->at(0,0,0,o));
  }
  printf("\n");
}

void test_fc_bp_cpu() {
  

  Session * session = Session::GetNewSession();
  session->gpu = false;
  size_t batch_size = 1;
  session->batch_size = batch_size;
  in_nodes = 2;
  h1_nodes = 3;
  out_nodes = 3;

  ConstInitializer<float> const_init(2.0, 1.0);
  FC<float> h1(in_nodes, h1_nodes, &const_init);
  FC<float> out(h1_nodes, out_nodes, &const_init);

  size_t in_dims[4] = {batch_size, 1, 1, in_nodes};
  size_t h1_dims[4];
  in.GetTopsDims({in_dims}, {h1_dims});
  size_t out_dims[4];
  h1.GetTopsDims({h1_dims}, {out_dims});


  Tensor<float>* in_tensor = Tensor<float>::CreateTensorCPU(in_dims);
  Tensor<float>* in_tensor_diff = Tensor<float>::CreateTensorCPU(in_dims);

  Tensor<float>* h1_tensor = Tensor<float>::CreateTensorCPU(h1_dims);
  Tensor<float>* h1_tensor_diff = Tensor<float>::CreateTensorCPU(h1_dims);

  Tensor<float>* out_tensor = Tensor<float>::CreateTensorCPU(out_dims);
  Tensor<float>* out_tensor_diff = Tensor<float>::CreateTensorCPU(out_dims);

  Tensor<float>* y_out = Tensor<float>::CreateTensorCPU(out_dims);

  // init in tensor
  in_tensor->at(0, 0, 0, 0) = 0;
  in_tensor->at(0, 0, 0, 0) = 1;

  // init out tensor
  y_out->at(0,0,0,0) = 0;
  y_out->at(0,0,0,1) = 1;
  y_out->at(0,0,0,2) = 0;


  h1.Forward({in_tensor}, {h1_tensor});
  out.Forward({h1_tensor}, {out_tensor});

  for(int i = 0; i < out_nodes; i++) {
    out_tensor_diff->at(0,0,0,i) = y_out->at(0,0,0,i) - out_tensor->at(0,0,0,i);
  }

  out.Backward({out_tensor}, {out_tensor_diff}, {h1_tensor}, {h1_tensor_diff});
  h1.Backward({h1_tensor}, {h1_tensor_diff}, {in_tensor}, {in_tensor_diff});

  printf("show layer activations\n");
  show_fc_tensor_cpu(h1_tensor);
  show_fc_tensor_cpu(out_tensor);

  printf("show activation diffs\n");
  show_fc_tensor_cpu(out_tensor_diff);
  show_fc_tensor_cpu(h1_tensor_diff);
  show_fc_tensor_cpu(in_tensor_diff);

  printf("show parameters diffs\n");
  show_fc_w_cpu(out.W_diff_);
  show_fc_b_cpu(out.b_diff_);

  show_fc_w_cpu(h1.W_diff_);
  show_fc_b_cpu(h1.b_diff_);

  show_fc_w_cpu(in.W_diff_);
  show_fc_b_cpu(in.b_diff_);  

  delete in_tensor;
  delete in_tensor_diff;
  delete h1_tensor;
  delete h1_tensor_diff;
  delete out_tensor;
  delete out_tensor_diff;
}



int main() {
  test_fc_bp_cpu();
}

