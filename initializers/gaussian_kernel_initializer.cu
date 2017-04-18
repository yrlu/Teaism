#ifndef GAUSSIAN_INITIALIZER_CUH_
#define GAUSSIAN_INITIALIZER_CUH_

#include "basics/initializer.hpp"
#include "utils/helper_cuda.h"
#include <cmath>
#include <vector>

#define PI (3.1415926535)


template<class Dtype>
class GaussianKernelInitializer: public Initializer<Dtype> {
private:
  const double sigma_;

public:
  GaussianKernelInitializer(double sigma): sigma_(sigma) {}
  void Initialize(Tensor<Dtype>* W, Tensor<Dtype>* b, bool gpu = true) const;

  __host__ __device__ static void InitGaussian(Tensor<Dtype> * W, Tensor<Dtype> *b, const double sigma_) {
    Dtype* w_data_array = W->GetDataPtr();
    const size_t * w_dims = W->GetDims();
    
    // W
    for(int i = 0; i < w_dims[2]; i++) {
      // input channel i ;
      for (int j = 0; j < w_dims[3]; j++) {
        // output channel j;

        // init kernel according to mu_ and sigma_;
        double x_mu_ = (w_dims[1]-1)/2;
        double y_mu_ = (w_dims[0]-1)/2;
        for (int x = 0; x < w_dims[1]; ++x) {
          for (int y = 0; y < w_dims[0]; ++y) {
            double g = exp(-0.5 * (pow((x - x_mu_) / sigma_, 2.0) + pow((y - y_mu_) / sigma_, 2.0))) / (2 * PI * sigma_ * sigma_);
            int idx[4] = {y, x, i, j};
            W->at(idx) = (Dtype)g;
          }
        }

        // normalize the filter
        Dtype sum = 0.0;
        for (int x = 0; x < w_dims[1]; ++x) {
          for (int y = 0; y < w_dims[0]; ++y) {
            int idx[4] = {y, x, i, j};
            sum = sum + W->at(idx);
          }
        }
        for (int x = 0; x < w_dims[1]; ++x) {
          for (int y = 0; y < w_dims[0]; ++y) {
            int idx[4] = {y, x, i, j};
            W->at(idx) /= sum;
          }
        }
      }
    }


    // b
    for (int i = 0; i < b->size(); i++) {
      b->GetDataPtr()[i] = 0;
    }
  }
};

template <class Dtype>
__global__ void InitializeGPU(Tensor<Dtype> * W, Tensor<Dtype> *b, const double sigma) {
  GaussianKernelInitializer<Dtype>::InitGaussian(W, b, sigma);
}

template <class Dtype>
void GaussianKernelInitializer<Dtype>::Initialize(Tensor<Dtype>* W, Tensor<Dtype>* b, bool gpu) const {
  if (gpu) {
    InitializeGPU<<<1, 1>>>(W, b, sigma_);
  } else {
    GaussianKernelInitializer<Dtype>::InitGaussian(W, b, sigma_);
  }
}



#endif // GAUSSIAN_INITIALIZER_CUH_