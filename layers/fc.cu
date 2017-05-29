#ifndef FC_LAYER_CUH_
#define FC_LAYER_CUH_

#include "initializers/const_initializer.cu"
#include "basics/layer.hpp"
#include "utils/utils.cu"

#define BLOCKDIM 32


template <class Dtype>
class FC: public Layer<Dtype> {
public:

  FC(size_t in_channels, size_t out_channels, Initializer<Dtype>* initializer = NULL);

  ~FC();

  void Forward(const std::vector<Tensor<Dtype>*> &bottoms, 
               const std::vector<Tensor<Dtype>*> &tops);

  void Backward(const std::vector<Tensor<Dtype>*> &tops,
                const std::vector<Tensor<Dtype>*> &tops_diff,
                const std::vector<Tensor<Dtype>*> &bottoms,
                const std::vector<Tensor<Dtype>*> &bottoms_diff);

  void GetTopsDims(const std::vector<size_t*> &bottoms_dims, 
                   const std::vector<size_t*> &tops_dims);

  void UpdateWb(Dtype lr);

  const size_t in_channels;
  const size_t out_channels;

  double lr;

  Tensor<Dtype>* W_;
  Tensor<Dtype>* b_;
  Tensor<Dtype>* W_diff_;
  Tensor<Dtype>* b_diff_;
private:
  const Initializer<Dtype>* initializer_;
  void InitParams();
  void InitDiffs();
};




namespace FCGPUKernels {
  template <class Dtype> 
  __global__ void ForwardGPUShared(Tensor<Dtype> * bottom, Tensor<Dtype> * top, Tensor<Dtype> * W_, Tensor<Dtype> * b_) {
    size_t n = bottom->GetDims()[0];
    size_t in_channels = bottom->GetDims()[3];
    size_t out_channels = top->GetDims()[3];
    int bi = (blockDim.x * blockIdx.x) + threadIdx.x; // batch idx
    int o = (blockDim.y * blockIdx.y) + threadIdx.y;  // output idi

    extern __shared__ Dtype s[];

    Dtype* in = s;
    Dtype* w = &s[in_channels*BLOCKDIM];

    for(int j = threadIdx.y; j < in_channels; j+= BLOCKDIM) {
      if(bi < n) {
        in[threadIdx.x*in_channels + j] = bottom->at(bi, 0, 0, j);
      }
    }

    for(int j = threadIdx.x; j < in_channels; j+= BLOCKDIM) {
      if(o < out_channels) {
        w[threadIdx.y*in_channels + j] = W_->at(0,0, blockDim.y * blockIdx.y + threadIdx.y, j);
      }
    }
    __syncthreads();

    if (bi < 0 || bi >= n || o < 0 || o >= out_channels) {
      return;
    }

    Dtype sum = 0;
    for(int i = 0; i < in_channels; i++) {
      sum += in[threadIdx.x*in_channels + i] * w[threadIdx.y*in_channels+i];
    }
    sum += b_->at(0,0,0,o);
    top->at(bi,0,0,o) = sum;
  }

  template <class Dtype>
  __global__ void ForwardGPU(Tensor<Dtype> * bottom, Tensor<Dtype> * top, Tensor<Dtype> * W_, Tensor<Dtype> * b_) {
    size_t n = bottom->GetDims()[0];
    size_t in_channels = bottom->GetDims()[3];
    size_t out_channels = top->GetDims()[3];
    int bi = (blockDim.x * blockIdx.x) + threadIdx.x; // batch idx
    int o = (blockDim.y * blockIdx.y) + threadIdx.y;  // output idi
    
    if (bi < 0 || bi >= n || o < 0 || o >= out_channels) {
      return;
    }
    Dtype sum = 0;
    for(int i = 0; i < in_channels; i++) {
      sum += bottom->at(bi, 0, 0, i) * W_->at(0,0,o,i);
      // sum += bottom->at(bi, 0, 0, i) * w[GetIdx(w_dims, 0, o, i)];
    }
    sum += b_->at(0,0,0,o);
    top->at(bi,0,0,o) = sum;
  }


  template<class Dtype>
  __global__ void ComputeBottomDiffs(Tensor<Dtype>* top_diff, Tensor<Dtype> * bottom_diff, Tensor<Dtype> * W_) {
    int b = (blockDim.x * blockIdx.x) + threadIdx.x;
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    
    size_t batch_size = bottom_diff->GetDims()[0];
    size_t in_channels = bottom_diff->GetDims()[3];
    size_t out_channels = top_diff->GetDims()[3];

    if(b < 0 || b >= batch_size || i < 0 || i >= in_channels) return;
    
    Dtype sum = 0;
    for(int o = 0; o < out_channels; o++) {
      sum += W_->at(0,0,o,i) * top_diff->at(b,0,0,o);
    }
    bottom_diff->at(b,0,0,i) = sum;
  }

  template<class Dtype>
  __global__ void ComputeWDiffs(Tensor<Dtype>* bottom, Tensor<Dtype> * top_diff, Tensor<Dtype> * W_diff_) {
    int i = (blockDim.x * blockIdx.x) + threadIdx.x;
    int o = (blockDim.y * blockIdx.y) + threadIdx.y;
    
    size_t batch_size = bottom->GetDims()[0];
    size_t in_channels = bottom->GetDims()[3];
    size_t out_channels = top_diff->GetDims()[3];

    if(i < 0 || i >= in_channels || o < 0 || o >= out_channels) return;

    Dtype sum_w = 0;
    for(int b = 0; b < batch_size; b++) {
      sum_w += bottom->at(b, 0, 0, i)*top_diff->at(b, 0, 0, o);
    }
    W_diff_->at(0,0,o,i) = sum_w;
  }

  template<class Dtype>
  __global__ void ComputeBDiffs(Tensor<Dtype> * top_diff, Tensor<Dtype>* b_diff_) {
    int o = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    size_t batch_size = top_diff->GetDims()[0];
    size_t out_ch = top_diff->GetDims()[3];

    if(o < 0 || o >= out_ch) return;

    Dtype sum_b = 0;
    for(int b = 0; b < batch_size; b++) {
      sum_b += top_diff->at(b, 0, 0, o);
    }
    b_diff_->at(0,0,0,o) = sum_b;
  }

  template<class Dtype>
  __global__ void UpdateWb(Tensor<Dtype> * W_, Tensor<Dtype> * W_diff_, Tensor<Dtype> * b_, Tensor<Dtype> * b_diff_, Dtype lr) {
    size_t in_channels = W_->GetDims()[3];
    size_t out_channels = W_->GetDims()[2];
    for(int o = 0; o < out_channels; o++) {
      for(int i = 0; i < in_channels; i++) {
        W_->at(0, 0, o, i) += W_diff_->at(0, 0, o, i)*lr; 
        W_diff_->at(0,0,o,i) = 0;
      }
      b_->at(0, 0, 0, o) += b_diff_->at(0, 0, 0, o)*lr;
      b_diff_->at(0,0,0,o) = 0;
    }
  }
}



template<class Dtype>
void FC<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) {
  assert(bottoms.size() == 1);
  assert(tops.size() == 1);
  Tensor<Dtype> * bottom = bottoms[0];
  Tensor<Dtype> * top = tops[0];

  if (Session::GetSession()->gpu) {
    size_t to_fc_dims[4];
    size_t b_dims[4];
    Tensor<Dtype>::GetTensorGPUDims(bottom, b_dims);
    to_fc_dims[0] = b_dims[0];
    to_fc_dims[1] = 1;
    to_fc_dims[2] = 1;
    to_fc_dims[3] = b_dims[1]*b_dims[2]*b_dims[3];
    Tensor<Dtype>::ReshapeTensorGPU(bottom, to_fc_dims);
    size_t batch_size = Session::GetSession()->batch_size;
    dim3 blocksInGrid(batch_size / BLOCKDIM + 1, out_channels / BLOCKDIM + 1);
    dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);

    if (in_channels*2 < 384) {
      FCGPUKernels::ForwardGPUShared<<<blocksInGrid,threadsPerBlock, 2*in_channels*BLOCKDIM*sizeof(Dtype)>>>(bottom, top, W_, b_);
    } else {
      FCGPUKernels::ForwardGPU<<<blocksInGrid,threadsPerBlock>>>(bottom, top, W_, b_);
    }

    Tensor<Dtype>::ReshapeTensorGPU(bottom, b_dims);
  } else {
    size_t to_fc_dims[4];
    size_t b_dims[4];
    Tensor<Dtype>::GetTensorCPUDims(bottom, b_dims);
    to_fc_dims[0] = bottom->GetDims()[0];
    to_fc_dims[1] = 1;
    to_fc_dims[2] = 1;
    to_fc_dims[3] = bottom->GetDims()[1]*bottom->GetDims()[2]*bottom->GetDims()[3];

    bottom->SetDims(to_fc_dims);

    for(int b = 0; b < bottom->GetDims()[0]; b++) {
      for(int o = 0; o < out_channels; o++) {
        Dtype sum = 0;
        for(int i = 0; i < in_channels; i++) {
          sum += bottom->at(b, 0, 0, i) * W_->at(0,0,o,i);
        }
        sum += b_->at(0,0,0,o);
        top->at(b,0,0,o) = sum;
      }
    }

    bottom->SetDims(b_dims);
  }

}


template<class Dtype>
void FC<Dtype>::GetTopsDims(const std::vector<size_t*> &bottoms_dims, 
                   const std::vector<size_t*> &tops_dims) {
  assert(bottoms_dims.size());
  assert(tops_dims.size());
  size_t * b_dims = bottoms_dims[0];
  size_t * t_dims = tops_dims[0];
  t_dims[0] = b_dims[0];
  t_dims[1] = 1;
  t_dims[2] = 1;
  t_dims[3] = out_channels;
}


template <class Dtype>
void FC<Dtype>::InitParams() {
  if (initializer_!=NULL) {
    initializer_->Initialize(W_, b_, Session::GetSession()->gpu);
  } else {
    ConstInitializer<Dtype>(1.0, 1.0).Initialize(W_, b_, Session::GetSession()->gpu);
  }
}

template <class Dtype>
FC<Dtype>::FC(size_t in_channels, size_t out_channels, Initializer<Dtype>* initializer):
    in_channels(in_channels), out_channels(out_channels), initializer_(initializer), W_diff_(NULL), b_diff_(NULL) {
  size_t w_dims[4] = {1, 1, out_channels, in_channels};
  size_t b_dims[4] = {1, 1, 1, out_channels};
  if (Session::GetSession()->gpu) {
    W_ = Tensor<Dtype>::CreateTensorGPU(w_dims);
    b_ = Tensor<Dtype>::CreateTensorGPU(b_dims);
  } else {
    W_ = Tensor<Dtype>::CreateTensorCPU(w_dims);
    b_ = Tensor<Dtype>::CreateTensorCPU(b_dims);
  }
  InitParams();
  lr = Session::GetSession()->lr;
}


template<class Dtype>
FC<Dtype>::~FC() {
  if (Session::GetSession()->gpu) {
    if (W_!= NULL) {
      cudaFree(W_);
      W_ = NULL;
    }
    if (b_ != NULL) {
      cudaFree(b_);
      b_ = NULL;
    }
    if (W_diff_ != NULL) {
      cudaFree(W_diff_);
      W_diff_ = NULL;
    }
    if (b_diff_ != NULL) {
      cudaFree(b_diff_);
      b_diff_ = NULL;
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
    if(W_diff_ != NULL) {
      delete W_diff_;
      W_diff_ = NULL;
    }
    if(b_diff_ != NULL) {
      delete b_diff_; 
      b_diff_ = NULL;
    }
  }
}


template<class Dtype>
void FC<Dtype>::InitDiffs() {
  if(Session::GetSession()->gpu) {
    if(W_diff_ == NULL) {
      size_t w_dims[4] = {1, 1, out_channels, in_channels};
      W_diff_ = Tensor<Dtype>::CreateTensorGPU(w_dims);
    }
    if(b_diff_ == NULL) {
      size_t b_dims[4] = {1, 1, 1, out_channels};
      b_diff_ = Tensor<Dtype>::CreateTensorGPU(b_dims);
    }
  } else {
    if(W_diff_ == NULL) {
      size_t w_dims[4] = {1, 1, out_channels, in_channels};
      W_diff_ = Tensor<Dtype>::CreateTensorCPU(w_dims);
    }
    if(b_diff_ == NULL) {
      size_t b_dims[4] = {1, 1, 1, out_channels};
      b_diff_ = Tensor<Dtype>::CreateTensorCPU(b_dims);
    }
  }
}



template<class Dtype>
void FC<Dtype>::UpdateWb(Dtype lr) {
  if(Session::GetSession()->gpu) {
    FCGPUKernels::UpdateWb<Dtype><<<1,1>>>(W_, W_diff_, b_, b_diff_, lr);
  } else {
    for(int o = 0; o < out_channels; o++) {
      for(int i = 0; i < in_channels; i++) {
        W_->at(0, 0, o, i) += W_diff_->at(0, 0, o, i)*lr; 
        W_diff_->at(0,0,o,i) = 0;
      }
      b_->at(0, 0, 0, o) += b_diff_->at(0, 0, 0, o)*lr;
      b_diff_->at(0,0,0,o) = 0;
    }
  }
}


template<class Dtype>
void FC<Dtype>::Backward(const std::vector<Tensor<Dtype>*> &tops,
                         const std::vector<Tensor<Dtype>*> &tops_diff,
                         const std::vector<Tensor<Dtype>*> &bottoms,
                         const std::vector<Tensor<Dtype>*> &bottoms_diff) {
  assert(bottoms.size() == 1);
  assert(bottoms_diff.size() == 1);
  assert(tops.size() == 1);
  assert(tops_diff.size() == 1);
  InitDiffs();
  size_t batch_size = Session::GetSession()->batch_size;
  Tensor<Dtype>* top = tops[0];
  Tensor<Dtype>* top_diff = tops_diff[0];
  Tensor<Dtype>* bottom = bottoms[0];
  Tensor<Dtype>* bottom_diff = bottoms_diff[0];

  if(Session::GetSession()->gpu) {

    size_t b_dims[4];
    Tensor<Dtype>::GetTensorGPUDims(bottom, b_dims);
    size_t to_fc_dims[4];
    to_fc_dims[0] = b_dims[0];
    to_fc_dims[1] = 1;
    to_fc_dims[2] = 1;
    to_fc_dims[3] = b_dims[1]*b_dims[2]*b_dims[3];
    Tensor<Dtype>::ReshapeTensorGPU(bottom, to_fc_dims);
    Tensor<Dtype>::ReshapeTensorGPU(bottom_diff, to_fc_dims);

    dim3 blocksInGrid(batch_size/BLOCKDIM+1, in_channels/BLOCKDIM+1);
    dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
    FCGPUKernels::ComputeBottomDiffs<Dtype><<<blocksInGrid, threadsPerBlock>>>(top_diff, bottom_diff, W_);
    blocksInGrid = dim3(in_channels/BLOCKDIM+1, out_channels/BLOCKDIM+1);
    FCGPUKernels::ComputeWDiffs<Dtype><<<blocksInGrid, threadsPerBlock>>>(bottom, top_diff, W_diff_);
    FCGPUKernels::ComputeBDiffs<Dtype><<<out_channels/BLOCKDIM/BLOCKDIM+1, BLOCKDIM*BLOCKDIM>>>(top_diff, b_diff_);

    // convert the shape back
    Tensor<Dtype>::ReshapeTensorGPU(bottom, b_dims);
    Tensor<Dtype>::ReshapeTensorGPU(bottom_diff, b_dims);
  } else {

    size_t b_dims[4];
    Tensor<Dtype>::GetTensorCPUDims(bottom, b_dims);
    size_t to_fc_dims[4];
    to_fc_dims[0] = b_dims[0];
    to_fc_dims[1] = 1;
    to_fc_dims[2] = 1;
    to_fc_dims[3] = b_dims[1]*b_dims[2]*b_dims[3];
    Tensor<Dtype>::ReshapeTensorCPU(bottom, to_fc_dims);
    Tensor<Dtype>::ReshapeTensorCPU(bottom_diff, to_fc_dims);


    // compute bottom diffs
    for(int i = 0; i < in_channels; i++) {
      for(int b = 0; b < batch_size; b++) {
        Dtype sum = 0;
        for(int o = 0; o < out_channels; o++) {
          sum += W_->at(0,0,o,i) * top_diff->at(b, 0, 0, o);
        }
        bottom_diff->at(b, 0, 0, i) = sum;
      }
    }
    
    // compute gradients for W, b
    for(int o = 0; o < out_channels; o++) {
      for(int i = 0; i < in_channels; i++) { 
        Dtype sum_w = 0;
        for(int b = 0; b < batch_size; b++) {
          sum_w += bottom->at(b, 0, 0, i)*top_diff->at(b, 0, 0, o);
        }
        W_diff_->at(0,0,o,i) = sum_w;
      }
    }
    for(int o = 0; o < out_channels; o++) {
      Dtype sum_b = 0;
      for(int b = 0; b < batch_size; b++) {
        sum_b += top_diff->at(b, 0, 0, o);
      }
      b_diff_->at(0,0,0,o) = sum_b; 
    }

    Tensor<Dtype>::ReshapeTensorCPU(bottom, b_dims);
    Tensor<Dtype>::ReshapeTensorCPU(bottom_diff, b_dims);
  }

  UpdateWb((Dtype)lr);
}

#endif  // FC_LAYER_CUH_
