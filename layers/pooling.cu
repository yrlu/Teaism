#ifndef POOLING_LAYER_CUH_
#define POOLING_LAYER_CUH_


enum POOLING_TYPE {MAX, MIN, AVERAGE};


template <class Dtype>
__global__ void pool(Tensor<Dtype> * bottom, Tensor<Dtype> * top, int bi, int o, size_t size, POOLING_TYPE type) {
  // bi is the index of the tensor
  // o is the output channel
  int x_top = (blockDim.x * blockIdx.x) + threadIdx.x;
  int y_top = (blockDim.y * blockIdx.y) + threadIdx.y;
  int x = x_top*size;
  int y = y_top*size;
  
  if (!bottom->isValidIdx(bi, y, x, o) || !top->isValidIdx(bi, y_top, x_top, o)) {
    return;
  }

  if (type==MAX) {
    Dtype pooled_val=INT_MIN;
    for(int i = y; i < y + size; i++) {
      for(int j = x; j < x + size; j++) {
        Dtype val = bottom->at(bi, i, j, o);
        if (val > pooled_val) {
          pooled_val = bottom->at(bi, i, j, o)
        }
      }
    }
    top->at(b, y_top, x_top, o) = pooled_val;
  } else if(type==MIN) {
    Dtype pooled_val=INT_MIN;
    for(int i = y; i < y + size; i++) {
      for(int j = x; j < x + size; j++) {
        Dtype val = bottom->at(bi, i, j, o);
        if (val < pooled_val) {
          pooled_val = bottom->at(bi, i, j, o)
        }
      }
    }
    top->at(b, y_top, x_top, o) = pooled_val;
  } else if(type==AVERAGE) {
    Dtype pooled_val=0;
    for(int i = y; i < y + size; i++) {
      for(int j = x; j < x + size; j++) {
        pooled_val += bottom->at(bi, i, j, o);
      }
    }
    top->at(b, y_top, x_top, o) = pooled_val/size/size;
  }
}


template <class Dtype>
__global__ void PoolingForwardGPU(Tensor<Dtype> * bottom, Tensor<Dtype> * top, size_t size, POOLING_TYPE type) {
  dim3 blocksInGrid(wid / BLOCKDIM + 1, hei / BLOCKDIM + 1);
  dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
  for (int b = 0; b < n; b++) {
    for (int o = 0; o < out_channels; o++) {
      pool<Dtype><<<blocksInGrid, threadsPerBlock>>>(bottom, top, b, o, size, type);
    }
  }
}

template <class Dtype>
class Pooling: public Layer<Dtype> {
public:
  Pooling(size_t size=2, POOLING_TYPE type=MAX):size_(size), type_(type) {}
  ~Pooling() {}

  void Forward(Tensor<Dtype> * bottom, Tensor<Dtype> * top) {
    if (Session::GetSession()->gpu) {
      PoolingForwardGPU<<<1,1>>>(bottom, top, size, type);
    } else {
      for(int b = 0; b < bottom->GetDims()[0]; b++) {
        for(int o = 0; o < bottom->GetDims()[3]; o++) {
          for(int x = 0, x_top = 0; x < bottom->GetDims()[2] && x_top < top->GetDims()[2]; x += size, x_top += 1) {
            for(int y = 0, y_top = 0; y < bottom->GetDims()[1] && y_top < top->GetDims()[1]; y += size, y_top += 1) {
              if (type==MAX) {
                Dtype pooled_val=INT_MIN;
                for(int i = y; i < y + size; i++) {
                  for(int j = x; j < x + size; j++) {
                    Dtype val = bottom->at(b, i, j, o);
                    if (val > pooled_val) {
                      pooled_val = bottom->at(b, i, j, o)
                    }
                  }
                }
                top->at(b, y_top, x_top, o) = pooled_val;
              } else if(type==MIN) {
                Dtype pooled_val=INT_MIN;
                for(int i = y; i < y + size; i++) {
                  for(int j = x; j < x + size; j++) {
                    Dtype val = bottom->at(b, i, j, o);
                    if (val < pooled_val) {
                      pooled_val = bottom->at(b, i, j, o)
                    }
                  }
                }
                top->at(b, y_top, x_top, o) = pooled_val;
              } else if(type==AVERAGE) {
                Dtype pooled_val=0;
                for(int i = y; i < y + size; i++) {
                  for(int j = x; j < x + size; j++) {
                    pooled_val += bottom->at(b, i, j, o);
                  }
                }
                top->at(b, y_top, x_top, o) = pooled_val/size/size;
              }
            }
          }
        }
      }
    }
  }

private:
  size_t size_;
  POOLING_TYPE type_;
};


#endif // POOLING_LAYER_CUH_
