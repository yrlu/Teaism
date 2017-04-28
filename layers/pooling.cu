#ifndef POOLING_LAYER_CUH_
#define POOLING_LAYER_CUH_

#include <assert.h>
#include "basics/layer.hpp"
#include "basics/tensor.cu"
#include "basics/session.hpp"

#define BLOCKDIM 32

enum POOLING_TYPE {MAX, MIN, AVERAGE};

namespace PoolingGPUKernels {

  template <class Dtype>
  __global__ void ForwardGPUKernel(Tensor<Dtype> * bottom, Tensor<Dtype> * top, int bi, int o, size_t size, POOLING_TYPE type, size_t stride) {
    // bi is the index of the tensor
    // o is the output channel
    int x_top = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y_top = (blockDim.y * blockIdx.y) + threadIdx.y;
    int x = x_top*stride;
    int y = y_top*stride;

    int hei = bottom->GetDims()[1];
    int wid = bottom->GetDims()[2];
    
    if (!bottom->isValidIdx(bi, y, x, o) || !top->isValidIdx(bi, y_top, x_top, o)) {
      return;
    }

    if (type==MAX) {
      Dtype pooled_val=bottom->at(bi, y, x, o);
      for(int i = y; i < y + size && i < hei; i++) {
        for(int j = x; j < x + size && j < wid; j++) {
          Dtype val = bottom->at(bi, i, j, o);
          if (val > pooled_val) {
            pooled_val = val;
          }
        }
      }
      top->at(bi, y_top, x_top, o) = pooled_val;
    } else if(type==MIN) {
      Dtype pooled_val=bottom->at(bi, y, x, o);
      for(int i = y; i < y + size && i < hei; i++) {
        for(int j = x; j < x + size && j < wid; j++) {
          Dtype val = bottom->at(bi, i, j, o);
          if (val < pooled_val) {
            pooled_val = val;
          }
        }
      }
      top->at(bi, y_top, x_top, o) = pooled_val;
    } else if(type==AVERAGE) {
      Dtype pooled_val=0;
      int cnt = 0;
      for(int i = y; i < y + size && i < hei; i++) {
        for(int j = x; j < x + size && j < wid; j++) {
          pooled_val += bottom->at(bi, i, j, o);
          cnt += 1;
        }
      }
      top->at(bi, y_top, x_top, o) = pooled_val/cnt;
    }
  }

  template <class Dtype>
  __global__ void ForwardGPU(Tensor<Dtype> * bottom, Tensor<Dtype> * top, size_t size, POOLING_TYPE type, size_t stride) {
    size_t n = bottom->GetDims()[0];
    size_t hei = top->GetDims()[1];
    size_t wid = top->GetDims()[2];
    size_t out_channels = top->GetDims()[3];

    dim3 blocksInGrid(wid / BLOCKDIM + 1, hei / BLOCKDIM + 1);
    dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
    for (int b = 0; b < n; b++) {
      for (int o = 0; o < out_channels; o++) {
        PoolingGPUKernels::ForwardGPUKernel<Dtype><<<blocksInGrid, threadsPerBlock>>>(bottom, top, b, o, size, type, stride);
      }
    }
  }

  template <class Dtype>
  __global__ void BackwardGPU(Tensor<Dtype>* top, Tensor<Dtype>* top_diff,
                              Tensor<Dtype>* bottom, Tensor<Dtype>* bottom_diff,
                              size_t size_, size_t stride_, POOLING_TYPE type_) {
    int batch_idx = threadIdx.x;
    bottom_diff->ResetVals(0);
    for(int o = 0; o < bottom->GetDims()[3]; o++) {
      for(int x = 0, x_top = 0; x < bottom->GetDims()[2] && x_top < top->GetDims()[2]; x += stride_, x_top += 1) {
        for(int y = 0, y_top = 0; y < bottom->GetDims()[1] && y_top < top->GetDims()[1]; y += stride_, y_top += 1) {
          int arg_idx[4] = {batch_idx, y, x, o};
          Dtype pooled_val = bottom->at(batch_idx, y, x, o);
          if (type_==MAX) {
            for(int i = y; i < y + size_ && i < bottom->GetDims()[1]; i++) {
              for(int j = x; j < x + size_ && j < bottom->GetDims()[2]; j++) {
                Dtype val = bottom->at(batch_idx, i, j, o);
                if (val > pooled_val) {
                  pooled_val = val;
                  arg_idx[0] = batch_idx;  arg_idx[1] = i;  arg_idx[2] = j;  arg_idx[3] = o;
                }
              }
            }
            bottom_diff->at(arg_idx) = top_diff->at(batch_idx, y_top, x_top, o);
          }
        }
      }
    }

  }

}

template <class Dtype>
class Pooling: public Layer<Dtype> {
public:
  Pooling(size_t size=2, POOLING_TYPE type=MAX, size_t stride=1):size_(size), type_(type), stride_(stride) {}
  ~Pooling() {}

  void GetTopsDims(const std::vector<size_t*> &bottoms_dims, 
                  const std::vector<size_t*> &tops_dims);

  void Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops);

  void Backward(const std::vector<Tensor<Dtype>*>&, const std::vector<Tensor<Dtype>*>&,
                const std::vector<Tensor<Dtype>*>&, const std::vector<Tensor<Dtype>*>&);

private:
  size_t size_;
  size_t stride_;
  POOLING_TYPE type_;
};


template<class Dtype>
void Pooling<Dtype>::GetTopsDims(const std::vector<size_t*> &bottoms_dims, 
                  const std::vector<size_t*> &tops_dims) {
  assert(bottoms_dims.size());
  assert(tops_dims.size());
  size_t * b_dims = bottoms_dims[0];
  size_t * t_dims = tops_dims[0];
  t_dims[0] = b_dims[0];
  t_dims[1] = b_dims[1]/stride_;
  t_dims[2] = b_dims[2]/stride_;
  t_dims[3] = b_dims[3];
}



template<class Dtype>
void Pooling<Dtype>::Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) {
  assert(bottoms.size()==1);
  assert(tops.size()==1);
  Tensor<Dtype> * bottom = bottoms[0];
  Tensor<Dtype> * top = tops[0];

  if (Session::GetSession()->gpu) {
    PoolingGPUKernels::ForwardGPU<<<1,1>>>(bottom, top, size_, type_, stride_);
  } else {
    for(int b = 0; b < bottom->GetDims()[0]; b++) {
      for(int o = 0; o < bottom->GetDims()[3]; o++) {
        for(int x = 0, x_top = 0; x < bottom->GetDims()[2] && x_top < top->GetDims()[2]; x += stride_, x_top += 1) {
          for(int y = 0, y_top = 0; y < bottom->GetDims()[1] && y_top < top->GetDims()[1]; y += stride_, y_top += 1) {
            if (type_==MAX) {
              Dtype pooled_val=bottom->at(b, y, x, o);
              for(int i = y; i < y + size_ && i < bottom->GetDims()[1]; i++) {
                for(int j = x; j < x + size_ && j < bottom->GetDims()[2]; j++) {
                  Dtype val = bottom->at(b, i, j, o);
                  if (val > pooled_val) {
                    pooled_val = val;
                  }
                }
              }
              top->at(b, y_top, x_top, o) = pooled_val;
            } else if(type_==MIN) {
              Dtype pooled_val=bottom->at(b, y, x, o);
              for(int i = y; i < y + size_ && i < bottom->GetDims()[1]; i++) {
                for(int j = x; j < x + size_ && j < bottom->GetDims()[2]; j++) {
                  Dtype val = bottom->at(b, i, j, o);
                  if (val < pooled_val) {
                    pooled_val = val;
                  }
                }
              }
              top->at(b, y_top, x_top, o) = pooled_val;
            } else if(type_==AVERAGE) {
              Dtype pooled_val=0;
              int cnt = 0;
              for(int i = y; i < y + size_ && i < bottom->GetDims()[1]; i++) {
                for(int j = x; j < x + size_ && j < bottom->GetDims()[2]; j++) {
                  pooled_val += bottom->at(b, i, j, o);
                  cnt ++;
                }
              }
              top->at(b, y_top, x_top, o) = pooled_val/cnt;
            }
          }
        }
      }
    }
  }
}

template <class Dtype>
void Pooling<Dtype>::Backward(const std::vector<Tensor<Dtype>*> &tops,
                              const std::vector<Tensor<Dtype>*> &tops_diff,
                              const std::vector<Tensor<Dtype>*> &bottoms,
                              const std::vector<Tensor<Dtype>*> &bottoms_diff) {
  assert(tops.size() == 1);
  assert(tops_diff.size() == 1);
  assert(bottoms.size() == 1);
  assert(bottoms_diff.size() == 1);

  Tensor<Dtype>* top = tops[0];
  Tensor<Dtype>* top_diff = tops_diff[0];
  Tensor<Dtype>* bottom = bottoms[0];
  Tensor<Dtype>* bottom_diff = bottoms_diff[0];

  Session* S = Session::GetSession();
  int batch_size = S->batch_size;
  if (S->gpu) {
    PoolingGPUKernels::BackwardGPU<Dtype><<<1,batch_size>>> 
      (top, top_diff, bottom, bottom_diff, size_, stride_, type_);
  } else {
    bottom_diff->ResetVals(0);
    for(int b = 0; b < bottom->GetDims()[0]; b++) {
      for(int o = 0; o < bottom->GetDims()[3]; o++) {
        for(int x = 0, x_top = 0; x < bottom->GetDims()[2] && x_top < top->GetDims()[2]; x += stride_, x_top += 1) {
          for(int y = 0, y_top = 0; y < bottom->GetDims()[1] && y_top < top->GetDims()[1]; y += stride_, y_top += 1) {
            int arg_idx[4] = {b, y, x, o};
            Dtype pooled_val = bottom->at(b, y, x, o);
            if (type_==MAX) {
              for(int i = y; i < y + size_ && i < bottom->GetDims()[1]; i++) {
                for(int j = x; j < x + size_ && j < bottom->GetDims()[2]; j++) {
                  Dtype val = bottom->at(b, i, j, o);
                  if (val > pooled_val) {
                    pooled_val = val;
                    arg_idx[0] = b;  arg_idx[1] = i;  arg_idx[2] = j;  arg_idx[3] = o;
                  }
                }
              }
              bottom_diff->at(arg_idx) = top_diff->at(b, y_top, x_top, o);
            }
          }
        }
      } 
    }  
  }
}


#endif // POOLING_LAYER_CUH_
