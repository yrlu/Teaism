#ifndef FC_LAYER_CUH_
#define FC_LAYER_CUH_



template <class Dtype>
class FC: public Layer<Dtype> {
public:

  FC(size_t in_channels, size_t out_channels, size_t stride, 
    Initializer<Dtype>* initializer = NULL):
      in_channels(in_channels), out_channels(out_channels), 
      stride(stride), initializer_(initializer) {
    size_t w_dims[4] = {1, 1, in_channels, out_channels};
    size_t b_dims[4] = {1, 1, in_channels, out_channels};
    if (Session::GetSession()->gpu) {
      W_ = Tensor<Dtype>::CreateTensorGPU(w_dims);
      b_ = Tensor<Dtype>::CreateTensorGPU(b_dims);
    } else {
      W_ = Tensor<Dtype>::CreateTensorCPU(w_dims);
      b_ = Tensor<Dtype>::CreateTensorCPU(b_dims);
    }
    InitParams();
  }

  ~FC() {
    if (Session::GetSession()->gpu) {
      if (W_!= NULL) {
        cudaFree(W_);
        W_ = NULL;
      }
      if (b_ != NULL) {
        cudaFree(b_);
        b_ = NULL;
      }
    } else {
      if(W_ != NULL) {
        delete W_;
        W_ = NULL;
      }
      if(b_ != NULL) {
        delete b_;
        b_ = NULL;
      }
    }
  }

  virtual void Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) {
    
  }

  const size_t in_channels;
  const size_t out_channels;
  const size_t stride;

private:
  Tensor<Dtype>* W_;
  Tensor<Dtype>* b_;
  const Initializer<Dtype>* initializer_;
  void InitParams() {
    if (initializer_!=NULL) {
      initializer_->Initialize(W_, b_, Session::GetSession()->gpu);
    } else {
      
      // GaussianKernelInitializer<Dtype>((Dtype)(kernel_width+kernel_height)/2).Initialize(W_, b_, Session::GetSession()->gpu);
    }
  }
};



#endif  // FC_LAYER_CUH_