#include <stdio.h>
#include <assert.h>
#include <vector>
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


__global__ void show_fc_tensor_gpu(Tensor<float> * fc_tensor_gpu) {
  for(int b = 0; b < fc_tensor_gpu->GetDims()[0]; b++) {
    for(int i = 0; i < fc_tensor_gpu->GetDims()[3]; i++) {
      printf("%f ", fc_tensor_gpu->at(b, 0, 0, i));
    }
    printf("\n");
  }
  printf("\n");
}


void show_fc_tensor_cpu(Tensor<float> * fc_tensor_cpu) {
  for(int b = 0; b < fc_tensor_cpu->GetDims()[0]; b++) {
    for(int i = 0; i < fc_tensor_cpu->GetDims()[3]; i++) {
      printf("%f ", fc_tensor_cpu->at(b, 0, 0, i));
    }
    printf("\n");
  }
  printf("\n");
}


__global__ void show_fc_w_gpu(Tensor<float> * fc_w_gpu) {
  for(int i = 0; i < fc_w_gpu->GetDims()[3]; i++) {
    for(int o = 0; o < fc_w_gpu->GetDims()[2]; o++) {
      printf("%f ", fc_w_gpu->at(0,0,o,i));
    }
    printf("\n");
  }
  printf("\n");
}

void show_fc_w_cpu(Tensor<float> * fc_w_cpu) {
  for(int i = 0; i < fc_w_cpu->GetDims()[3]; i++) {
    for(int o = 0; o < fc_w_cpu->GetDims()[2]; o++) {
      printf("%f ", fc_w_cpu->at(0,0,o,i));
    }
    printf("\n");
  }
  printf("\n");
}

__global__ void show_fc_b_gpu(Tensor<float> * fc_b_gpu) {
  for(int o = 0; o < fc_b_gpu->GetDims()[3]; o++) {
    printf("%f ", fc_b_gpu->at(0,0,0,o));
  }
  printf("\n");
}

void show_fc_b_cpu(Tensor<float> * fc_b_cpu) {
  for(int o = 0; o < fc_b_cpu->GetDims()[3]; o++) {
    printf("%f ", fc_b_cpu->at(0,0,0,o));
  }
  printf("\n");
}

void test_fc_bp_cpu() {
  // The example shows counting how many ones in the input:
  // {0,0} -> {0,0,1}
  // {0,1} -> {0,1,0}
  // {1,0} -> {0,1,0}
  // {1,1} -> {1,0,0}

  Session * session = Session::GetNewSession();
  session->gpu = false;
  size_t batch_size = 1;
  session->batch_size = batch_size;
  size_t in_nodes = 2;
  size_t h1_nodes = 3;
  size_t out_nodes = 3;

  ConstInitializer<float> const_init(2.0, 1.0);
  FC<float> h1(in_nodes, h1_nodes, &const_init);
  FC<float> out(h1_nodes, out_nodes, &const_init);

  size_t in_dims[4] = {batch_size, 1, 1, in_nodes};
  size_t h1_dims[4];
  h1.GetTopsDims({in_dims}, {h1_dims});
  size_t out_dims[4];
  out.GetTopsDims({h1_dims}, {out_dims});

  Tensor<float>* in_tensor = Tensor<float>::CreateTensorCPU(in_dims);
  Tensor<float>* in_tensor_diff = Tensor<float>::CreateTensorCPU(in_dims);

  Tensor<float>* h1_tensor = Tensor<float>::CreateTensorCPU(h1_dims);
  Tensor<float>* h1_tensor_diff = Tensor<float>::CreateTensorCPU(h1_dims);

  Tensor<float>* out_tensor = Tensor<float>::CreateTensorCPU(out_dims);
  Tensor<float>* out_tensor_diff = Tensor<float>::CreateTensorCPU(out_dims);

  Tensor<float>* y_out = Tensor<float>::CreateTensorCPU(out_dims);

  std::vector<std::vector<float>> x_train = {{0,1},{0,0},{1,0},{1,1}};
  std::vector<std::vector<float>> y_train = {{0,1,0}, {0,0,1}, {0,1,0}, {1,0,0}};

  for(int iter = 0; iter < 2000; iter++) {
    
    // init in tensor
    in_tensor->at(0, 0, 0, 0) = x_train[iter%x_train.size()][0];
    in_tensor->at(0, 0, 0, 1) = x_train[iter%x_train.size()][1];

    // init out tensor
    y_out->at(0,0,0,0) = y_train[iter%y_train.size()][0];
    y_out->at(0,0,0,1) = y_train[iter%y_train.size()][1];
    y_out->at(0,0,0,2) = y_train[iter%y_train.size()][2];


    printf("\n-----iteration %d-------\n", iter);
    printf("input: %f %f \n", x_train[iter%x_train.size()][0], x_train[iter%x_train.size()][1]);

    h1.Forward({in_tensor}, {h1_tensor});
    out.Forward({h1_tensor}, {out_tensor});

    for(int i = 0; i < out_nodes; i++) {
      out_tensor_diff->at(0,0,0,i) = y_out->at(0,0,0,i) - out_tensor->at(0,0,0,i);
    }

    out.Backward({out_tensor}, {out_tensor_diff}, {h1_tensor}, {h1_tensor_diff});
    h1.Backward({h1_tensor}, {h1_tensor_diff}, {in_tensor}, {in_tensor_diff});

    printf("out activations\n");
    show_fc_tensor_cpu(out_tensor);

    /*  
    printf("h1 activations\n");
    show_fc_tensor_cpu(h1_tensor);
    
    printf("out activation diffs\n");
    show_fc_tensor_cpu(out_tensor_diff);
    printf("h1 activations diffs\n");
    show_fc_tensor_cpu(h1_tensor_diff);
    printf("in activations diffs\n");
    show_fc_tensor_cpu(in_tensor_diff);

    printf("out W diffs\n");
    show_fc_w_cpu(out.W_diff_);
    printf("out b diffs\n");
    show_fc_b_cpu(out.b_diff_);

    printf("h1 W diffs\n");
    show_fc_w_cpu(h1.W_diff_);
    printf("h1 b diffs\n");
    show_fc_b_cpu(h1.b_diff_);
    */

    printf("mse: %f \n", (out_tensor_diff->at(0,0,0,0)*out_tensor_diff->at(0,0,0,0) + out_tensor_diff->at(0,0,0,1)*out_tensor_diff->at(0,0,0,1) + out_tensor_diff->at(0,0,0,2)*out_tensor_diff->at(0,0,0,2))/3);
    out.UpdateWb(0.02);
    h1.UpdateWb(0.02);
  }


  delete in_tensor;
  delete in_tensor_diff;
  delete h1_tensor;
  delete h1_tensor_diff;
  delete out_tensor;
  delete out_tensor_diff;
}


void test_fc_bp_cpu2() {
  // The example shows counting how many ones in the input:
  // {0,0} -> {0,0,1}
  // {0,1} -> {0,1,0}
  // {1,0} -> {0,1,0}
  // {1,1} -> {1,0,0}
  
  Session * session = Session::GetNewSession();
  session->gpu = false;
  size_t batch_size = 4;
  session->batch_size = batch_size;
  size_t in_nodes = 2;
  size_t h1_nodes = 3;
  size_t out_nodes = 3;

  ConstInitializer<float> const_init(2.0, 1.0);
  FC<float> h1(in_nodes, h1_nodes, &const_init);
  FC<float> out(h1_nodes, out_nodes, &const_init);
  Softmax<float> softmax_layer;

  size_t in_dims[4] = {batch_size, 1, 1, in_nodes};
  size_t h1_dims[4];
  h1.GetTopsDims({in_dims}, {h1_dims});
  size_t out_dims[4];
  out.GetTopsDims({h1_dims}, {out_dims});
  size_t softmax_out_dims[4];
  softmax_layer.GetTopsDims({out_dims}, {softmax_out_dims});

  Tensor<float>* in_tensor = Tensor<float>::CreateTensorCPU(in_dims);
  Tensor<float>* in_tensor_diff = Tensor<float>::CreateTensorCPU(in_dims);

  Tensor<float>* h1_tensor = Tensor<float>::CreateTensorCPU(h1_dims);
  Tensor<float>* h1_tensor_diff = Tensor<float>::CreateTensorCPU(h1_dims);

  Tensor<float>* out_tensor = Tensor<float>::CreateTensorCPU(out_dims);
  Tensor<float>* out_tensor_diff = Tensor<float>::CreateTensorCPU(out_dims);
  Tensor<float>* softmax_out_tensor = Tensor<float>::CreateTensorCPU(softmax_out_dims);

  Tensor<float>* y_out = Tensor<float>::CreateTensorCPU(out_dims);

  std::vector<std::vector<float>> x_train = {{0,1},{0,0},{1,0},{1,1}};
  std::vector<std::vector<float>> y_train = {{0,1,0}, {0,0,1}, {0,1,0}, {1,0,0}};

  for(int b = 0; b < batch_size; b++) {
    // init in tensor
    in_tensor->at(b, 0, 0, 0) = x_train[b%x_train.size()][0];
    in_tensor->at(b, 0, 0, 1) = x_train[b%x_train.size()][1];

    // init out tensor
    y_out->at(b,0,0,0) = y_train[b%y_train.size()][0];
    y_out->at(b,0,0,1) = y_train[b%y_train.size()][1];
    y_out->at(b,0,0,2) = y_train[b%y_train.size()][2];
  }

  for(int iter = 0; iter < 2000; iter++) {
    
    printf("\n-----iteration %d-------\n", iter);
    // printf("input: %f %f \n", x_train[iter%x_train.size()][0], x_train[iter%x_train.size()][1]);

    h1.Forward({in_tensor}, {h1_tensor});
    out.Forward({h1_tensor}, {out_tensor});
    softmax_layer.Forward({out_tensor}, {softmax_out_tensor});

    for(int b = 0; b < batch_size; b++) {
      for(int i = 0; i < out_nodes; i++) {
        out_tensor_diff->at(b,0,0,i) = y_out->at(b,0,0,i) - out_tensor->at(b,0,0,i);
      }
    }

    out.Backward({out_tensor}, {out_tensor_diff}, {h1_tensor}, {h1_tensor_diff});
    h1.Backward({h1_tensor}, {h1_tensor_diff}, {in_tensor}, {in_tensor_diff});

    printf("out activations\n");
    show_fc_tensor_cpu(softmax_out_tensor);

    /*  
    printf("h1 activations\n");
    show_fc_tensor_cpu(h1_tensor);
    
    printf("out activation diffs\n");
    show_fc_tensor_cpu(out_tensor_diff);
    printf("h1 activations diffs\n");
    show_fc_tensor_cpu(h1_tensor_diff);
    printf("in activations diffs\n");
    show_fc_tensor_cpu(in_tensor_diff);

    printf("out W diffs\n");
    show_fc_w_cpu(out.W_diff_);
    printf("out b diffs\n");
    show_fc_b_cpu(out.b_diff_);

    printf("h1 W diffs\n");
    show_fc_w_cpu(h1.W_diff_);
    printf("h1 b diffs\n");
    show_fc_b_cpu(h1.b_diff_);
    */

    float mse = 0;
    for(int b = 0; b < batch_size; b++) {
      mse += (out_tensor_diff->at(b,0,0,0)*out_tensor_diff->at(b,0,0,0) + out_tensor_diff->at(b,0,0,1)*out_tensor_diff->at(b,0,0,1) + out_tensor_diff->at(b,0,0,2)*out_tensor_diff->at(b,0,0,2))/3;
    }
    mse /= batch_size;
    printf("mse: %f \n", mse);
    out.UpdateWb(0.01);
    h1.UpdateWb(0.01);
  }


  delete in_tensor;
  delete in_tensor_diff;
  delete h1_tensor;
  delete h1_tensor_diff;
  delete out_tensor;
  delete out_tensor_diff;
  delete softmax_out_tensor;
}

__global__ void calc_out_diff(Tensor<float> * out_tensor_diff, Tensor<float> * out_tensor, Tensor<float> * y_out) {
  size_t batch_size = y_out->GetDims()[0];
  size_t out_nodes = y_out->GetDims()[3];
  for(int b = 0; b < batch_size; b++) {
    for(int i = 0; i < out_nodes; i++) {
      out_tensor_diff->at(b,0,0,i) = y_out->at(b,0,0,i) - out_tensor->at(b,0,0,i);
    }
  }
}


__global__ void prepare_training_data(Tensor<float> *in_tensor, Tensor<float> * y_out) {


  in_tensor->at(0, 0, 0, 0) = 0;
  in_tensor->at(0, 0, 0, 1) = 1;

  y_out->at(0,0,0,0) = 0;
  y_out->at(0,0,0,1) = 1;
  y_out->at(0,0,0,2) = 0;


  in_tensor->at(1, 0, 0, 0) = 0;
  in_tensor->at(1, 0, 0, 1) = 0;

  y_out->at(1,0,0,0) = 0;
  y_out->at(1,0,0,1) = 0;
  y_out->at(1,0,0,2) = 1;


  in_tensor->at(2, 0, 0, 0) = 1;
  in_tensor->at(2, 0, 0, 1) = 0;

  y_out->at(2,0,0,0) = 0;
  y_out->at(2,0,0,1) = 1;
  y_out->at(2,0,0,2) = 0;


  in_tensor->at(3, 0, 0, 0) = 1;
  in_tensor->at(3, 0, 0, 1) = 1;

  y_out->at(3,0,0,0) = 1;
  y_out->at(3,0,0,1) = 0;
  y_out->at(3,0,0,2) = 0;
}

void test_fc_bp_gpu() {
  // The example shows counting how many ones in the input:
  // {0,0} -> {0,0,1}
  // {0,1} -> {0,1,0}
  // {1,0} -> {0,1,0}
  // {1,1} -> {1,0,0}

  cudaError_t cudaStatus = cudaSetDevice(0);
  checkCudaErrors(cudaStatus);
  
  Session * session = Session::GetNewSession();
  session->gpu = true;
  size_t batch_size = 4;
  session->batch_size = batch_size;
  size_t in_nodes = 2;
  size_t h1_nodes = 3;
  size_t out_nodes = 3;

  ConstInitializer<float> const_init(2.0, 1.0);
  FC<float> h1(in_nodes, h1_nodes, &const_init);
  FC<float> out(h1_nodes, out_nodes, &const_init);
  Softmax<float> softmax_layer;
  CrossEntropyLoss<float> cel_layer;

  size_t in_dims[4] = {batch_size, 1, 1, in_nodes};
  size_t h1_dims[4];
  h1.GetTopsDims({in_dims}, {h1_dims});
  size_t out_dims[4];
  out.GetTopsDims({h1_dims}, {out_dims});
  size_t softmax_out_dims[4];
  softmax_layer.GetTopsDims({out_dims}, {softmax_out_dims});
  size_t ce_out_dims[4];
  cel_layer.GetTopsDims({out_dims}, {ce_out_dims});


  Tensor<float>* in_tensor = Tensor<float>::CreateTensorGPU(in_dims);
  Tensor<float>* in_tensor_diff = Tensor<float>::CreateTensorGPU(in_dims);

  Tensor<float>* h1_tensor = Tensor<float>::CreateTensorGPU(h1_dims);
  Tensor<float>* h1_tensor_diff = Tensor<float>::CreateTensorGPU(h1_dims);

  Tensor<float>* out_tensor = Tensor<float>::CreateTensorGPU(out_dims);
  Tensor<float>* out_tensor_diff = Tensor<float>::CreateTensorGPU(out_dims);
  
  Tensor<float>* softmax_out_tensor = Tensor<float>::CreateTensorGPU(softmax_out_dims);
  Tensor<float>* softmax_out_tensor_diff = Tensor<float>::CreateTensorGPU(softmax_out_dims);

  Tensor<float>* cel_out_tensor = Tensor<float>::CreateTensorGPU(ce_out_dims);
  Tensor<float>* cel_out_tensor_diff = Tensor<float>::CreateTensorGPU(ce_out_dims);

  Tensor<float>* y_out = Tensor<float>::CreateTensorGPU(out_dims);

  std::vector<std::vector<float>> x_train = {{0,1},{0,0},{1,0},{1,1}};
  std::vector<std::vector<float>> y_train = {{0,1,0}, {0,0,1}, {0,1,0}, {1,0,0}};

  prepare_training_data<<<1,1>>>(in_tensor, y_out);

  for(int iter = 0; iter < 5000; iter++) {
    
    printf("\n-----iteration %d-------\n", iter);
    // printf("input: %f %f \n", x_train[iter%x_train.size()][0], x_train[iter%x_train.size()][1]);

    h1.Forward({in_tensor}, {h1_tensor});
    out.Forward({h1_tensor}, {out_tensor});
    // softmax_layer.Forward({out_tensor}, {softmax_out_tensor});
    cel_layer.Forward({out_tensor}, {cel_out_tensor});

    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);

    // calc_out_diff<<<1,1>>>(out_tensor_diff, out_tensor, y_out);
    // calc_out_diff<<<1,1>>>(softmax_out_tensor_diff, softmax_out_tensor, y_out);
    calc_out_diff<<<1,1>>>(cel_out_tensor_diff, cel_out_tensor, y_out);
    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);

    // softmax_layer.Backward({softmax_out_tensor}, {softmax_out_tensor_diff}, {out_tensor}, {out_tensor_diff});
    cel_layer.Backward({cel_out_tensor}, {cel_out_tensor_diff}, {out_tensor}, {out_tensor_diff});
    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);

    out.Backward({out_tensor}, {out_tensor_diff}, {h1_tensor}, {h1_tensor_diff});
    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);

    h1.Backward({h1_tensor}, {h1_tensor_diff}, {in_tensor}, {in_tensor_diff});

    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);
 
    printf("out activations\n");
    show_fc_tensor_gpu<<<1,1>>>(softmax_out_tensor);
  
    cudaStatus = cudaDeviceSynchronize();
    checkCudaErrors(cudaStatus);
   
    /*  
    printf("h1 activations\n");
    show_fc_tensor_gpu<<<1,1>>>(h1_tensor);
    
    printf("out activation diffs\n");
    show_fc_tensor_gpu<<<1,1>>>(out_tensor_diff);
    printf("h1 activations diffs\n");
    show_fc_tensor_gpu<<<1,1>>>(h1_tensor_diff);
    printf("in activations diffs\n");
    show_fc_tensor_gpu<<<1,1>>>(in_tensor_diff);

    printf("out W diffs\n");
    show_fc_w_gpu<<<1,1>>>(out.W_diff_);
    printf("out b diffs\n");
    show_fc_b_gpu<<<1,1>>>(out.b_diff_);

    printf("h1 W diffs\n");
    show_fc_w_gpu<<<1,1>>>(h1.W_diff_);
    printf("h1 b diffs\n");
    show_fc_b_gpu<<<1,1>>>(h1.b_diff_);
    */

    // float mse = 0;
    // for(int b = 0; b < batch_size; b++) {
    //   mse += (out_tensor_diff->at(b,0,0,0)*out_tensor_diff->at(b,0,0,0) + out_tensor_diff->at(b,0,0,1)*out_tensor_diff->at(b,0,0,1) + out_tensor_diff->at(b,0,0,2)*out_tensor_diff->at(b,0,0,2))/3;
    // }
    // mse /= batch_size;
    // printf("mse: %f \n", mse);
    out.UpdateWb(0.01);
    h1.UpdateWb(0.01);
  }


  cudaFree(in_tensor);
  cudaFree(in_tensor_diff);
  cudaFree(h1_tensor);
  cudaFree(h1_tensor_diff);
  cudaFree(out_tensor);
  cudaFree(out_tensor_diff);
  cudaFree(softmax_out_tensor);
  cudaFree(softmax_out_tensor_diff);
  cudaFree(cel_out_tensor);
  cudaFree(cel_out_tensor_diff);  
}


int main() {
  // test_fc_bp_cpu2();
  test_fc_bp_gpu();
}

