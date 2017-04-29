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



void test_fc_bp_cpu() {
  

  Session * session = Session::GetNewSession();
  session->gpu = false;
  session->batch_size = 1;

  ConstInitializer<float> const_init(2.0, 1.0);
  FC<float> in(2, 3, &const_init);
  FC<float> h1(3, 3, &const_init);
  
  size_t in_dims[4] = {}
  
}



int main() {
  test_fc_bp_cpu();
}

