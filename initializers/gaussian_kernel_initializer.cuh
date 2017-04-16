#ifndef GAUSSIAN_INITIALIZER_CUH_
#define GAUSSIAN_INITIALIZER_CUH_

#include "basics/initializer.hpp"
#include <cmath>

#define PI (3.1415926535)

// Initialize gaussian kernels
template<class Dtype>
class GaussianKernelInitializer: public Initializer<Dtype> {
private:
  const double sigma_;

  void InitGaussian(Tensor<Dtype>* W, Tensor<Dtype> *b) {
    // CPU
    Dtype* w_data_array = W->GetDataPtr();
    auto w_dims = W->GetDims();
    assert(w_dims.size() == 4);
    
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
            W->at({y, x, i, j}) = (Dtype)g;
          }
        }

        // normalize the filter
        Dtype sum = 0.0;
        for (int x = 0; x < w_dims[1]; ++x) {
          for (int y = 0; y < w_dims[0]; ++y) {
            sum += W->at({y, x, i, j});
          }
        }
        for (int x = 0; x < w_dims[1]; ++x) {
          for (int y = 0; y < w_dims[0]; ++y) {
            W->at({y, x, i, j}) /= sum;
          }
        }
      }
    }

    assert(b->GetDims().size() == 1);
    for (int i = 0; i < b->size(); i++) {
      b->GetDataPtr()[i] = 0;
    }
  }

public:
  GaussianKernelInitializer(double sigma): sigma_(sigma) {}
  void Initialize(Tensor<Dtype>* W, Tensor<Dtype>* b) const {
    if (W->gpu) {
      // TODO: GPU
      Dtype * w_data_array = W->GetDataPtr();
      auto w_dims = W->GetDims();
      assert(w_dims.size() == 4);
      
      Tensor<Dtype> * _W = new Tensor<Dtype>(W->GetDims());
      Tensor<Dtype> * _b = new Tensor<Dtype>(b->GetDims());
      InitGaussian(_W, _b);

      
  // cudaStatus = cudaMemcpy(*d_gaussian, &gaussian[0], FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  // checkCudaErrors(cudaStatus);
    } else {
      InitGaussian(W, b);
    }
  }

};

#endif // GAUSSIAN_INITIALIZER_CUH_