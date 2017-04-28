#ifndef COMPUTATIONS_CUH_
#define COMPUTATIONS_CUH_

#include "basics/tensor.cu"
#include "basics/commons.hpp"

#define BLOCKDIM 32

namespace ConvGPUKernels {

    template <class Dtype>
  __global__ void ForwardGPUKernel2(Tensor<Dtype> * bottom, Tensor<Dtype> * top, Tensor<Dtype> * W_, Tensor<Dtype> * b_, int hs, int ws, int stride, PADDING padding) {
    int bi = blockIdx.y/hs;
    int o = blockIdx.x/ws;

    int hi = blockIdx.y % hs;
    int wi = blockIdx.x % ws;

    int y_top = hi * BLOCKDIM + threadIdx.y;
    int x_top = wi * BLOCKDIM + threadIdx.x;

    if(!top->isValidIdx(bi, y_top, x_top, o)) return;

    size_t in_channels = bottom->GetDims()[3];
    size_t kernel_height = W_->GetDims()[0];
    size_t kernel_width = W_->GetDims()[1];
    
    int y = y_top * stride + kernel_height/2*(padding==VALID);
    int x = x_top * stride + kernel_width/2*(padding==VALID);

    extern __shared__ Dtype s[];
    Dtype * k = s;
    const size_t * w_dims = W_->GetDims();
    if(threadIdx.x < kernel_width && threadIdx.y < kernel_height) {
      for(int c = 0; c < in_channels; c++) {
        k[GetIdx(w_dims, threadIdx.y, threadIdx.x, c)] = W_->at(threadIdx.y, threadIdx.x, c, o);
      }
    }
    __syncthreads();


    Dtype sum = 0.0;
    for(int i = 0; i < kernel_height; i++) {
      for(int j = 0; j < kernel_width; j++) {
        for(int c = 0; c < in_channels; c++) {
          // (n, hei, wid, out_channelsnnel),   // (hei, wid, input, output)
          // sum += bottom->atPadding(bi, y+i-int(kernel_height/2), x+j-int(kernel_width/2), c) * W_->at(i, j, c, o);
          sum += bottom->atPadding(bi, y+i-int(kernel_height/2), x+j-int(kernel_width/2), c) * k[GetIdx(w_dims, i, j, c)];
        }
      }
    }
    if(b_!=NULL) {
      sum += b_->at(0,0,0,o);
    }
    top->at(bi, y_top, x_top, o) = sum;
  }

  template <class Dtype>
  __global__ void ForwardGPUKernel(Tensor<Dtype> * bottom, Tensor<Dtype> * top, Tensor<Dtype> * W_, Tensor<Dtype> * b_, int stride, PADDING padding) {
    size_t bs = bottom->GetDims()[0];
    size_t kernel_height = W_->GetDims()[0];
    size_t kernel_width = W_->GetDims()[1]; 
    size_t in_channels = bottom->GetDims()[3];
    size_t out_channels = top->GetDims()[3];

    int b = (blockDim.x * blockIdx.x) + threadIdx.x;
    int o = (blockDim.y * blockIdx.y) + threadIdx.y;
    if(b<0 || b>=bs || o<0 || o>=out_channels) {
      return;
    }
    
    if(padding==SAME) {
      for(int x = 0, x_top = 0; /*x < bottom->GetDims()[2]*/ x_top < top->GetDims()[2]; x += stride, x_top += 1) {
        for(int y = 0, y_top = 0; /*y < bottom->GetDims()[1]*/ y_top < top->GetDims()[1]; y += stride, y_top += 1) {
          int idx[4] = {b, y, x, o};
          Dtype sum = 0.0;
          for(int c = 0; c < in_channels; c++) {
            for(int i = 0; i < kernel_height; i++) {
              for(int j = 0; j < kernel_width; j++) {
                // (n, hei, wid, out_channelsnnel),   // (hei, wid, input, output)
                int b_idx[4] = {b, y+i-int(kernel_height/2), x+j-int(kernel_width/2), c};
                int t_idx[4] = {i, j, c, o};
                sum += bottom->atPadding(b_idx) * W_->at(t_idx);
              }
            }
          }
          if(b_!=NULL) {
            sum += b_->at(0,0,0,o);
          }
          top->at(b, y_top, x_top, o) = sum;
        }
      }
    } else if (padding==VALID) {
      for(int x = kernel_width/2, x_top = 0; /*x < bottom->GetDims()[2] - kernel_width/2*/ x_top < top->GetDims()[2]; x += stride, x_top += 1) {
        for(int y = kernel_height/2, y_top = 0; /*y < bottom->GetDims()[1] - kernel_height/2*/ y_top < top->GetDims()[1]; y += stride, y_top += 1) {
          // batch idx b, output layer o, pixel (x, y)
          // top->at({b, y, x, o}) = 
          int idx[4] = {b, y, x, o};
          Dtype sum = 0.0;
          for(int c = 0; c < in_channels; c++) {
            for(int i = 0; i < kernel_height; i++) {
              for(int j = 0; j < kernel_width; j++) {
                // (n, hei, wid, channel),   // (hei, wid, input, output)
                int b_idx[4] = {idx[0], idx[1]+i-int(kernel_height/2), idx[2]+j-int(kernel_width/2), c};
                int t_idx[4] = {i, j, c, idx[3]};
                sum += bottom->atPadding(b_idx) * W_->at(t_idx);
              }
            }
          }
          if (b_ != NULL) {
            sum += b_->at(0,0,0,o);
          }
          top->at(b, y_top, x_top, o) = sum;
        }
      }
    }
  }
}

namespace ComputationsGPU {

template<class Dtype>
__host__ void ConvolutionGPU(Tensor<Dtype> * in, Tensor<Dtype> * out, Tensor<Dtype> * W_, Tensor<Dtype> * b_, const size_t stride, PADDING padding) {

  size_t out_dims[4];
  size_t in_dims[4];
  size_t w_dims[4];
  Tensor<Dtype>::GetTensorGPUDims(in, in_dims);
  Tensor<Dtype>::GetTensorGPUDims(out, out_dims);
  Tensor<Dtype>::GetTensorGPUDims(W_, w_dims);

  size_t kernel_height = w_dims[0];
  size_t kernel_width = w_dims[1];
  size_t bs = in_dims[0];
  size_t hei = out_dims[1];
  size_t wid = out_dims[2];
  size_t in_channels = in_dims[3];
  size_t out_channels = out_dims[3];

  size_t hs = hei/BLOCKDIM + 1;
  size_t ws = wid/BLOCKDIM + 1;

  dim3 blocksInGrid(ws*out_channels, hs*bs);
  dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
  ConvGPUKernels::ForwardGPUKernel2<Dtype><<<blocksInGrid, threadsPerBlock, kernel_height*kernel_width*in_channels*sizeof(Dtype)>>>(in, out, W_, b_, hs, ws, stride, padding);
  

  // 2D Parallelization
  /*
  size_t bs = Session::GetSession()->batch_size;
  dim3 blocksInGrid(bs / BLOCKDIM + 1, out_channels / BLOCKDIM + 1);
  dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
  ConvGPUKernels::ForwardGPUKernel<Dtype><<<blocksInGrid, threadsPerBlock>>>(bottom, top, W_, b_, stride, padding);
  */
}

}





// CPU functions
namespace ComputationsCPU {

  template<class Dtype>
  __host__ void ConvolutionCPU(Tensor<Dtype> * in, Tensor<Dtype> * out, Tensor<Dtype> * W_, Tensor<Dtype> * b_, const size_t stride, PADDING padding) {
    size_t in_channels = in->GetDims()[3];
    size_t out_channels = out->GetDims()[3];
    size_t kernel_height = W_->GetDims()[0];
    size_t kernel_width = W_->GetDims()[1];   

    for(int b = 0; b < in->GetDims()[0]; b++) {
      for(int o = 0; o < out_channels; o++) {
        if(padding==SAME) {
          for(int x = 0, x_top = 0; x_top < out->GetDims()[2]; x += stride, x_top += 1) {
            for(int y = 0, y_top = 0; y_top < out->GetDims()[1]; y += stride, y_top += 1) {
              // batch idx b, output layer o, pixel (x, y)
              Dtype sum = 0.0;
              for(int c = 0; c < in_channels; c++) {
                for(int i = 0; i < kernel_height; i++) {
                  for(int j = 0; j < kernel_width; j++) {
                    // (n, hei, wid, channel),   // (hei, wid, input, output)
                    int b_idx[4] = {b, y+i-int(kernel_height/2), x+j-int(kernel_width/2), c};
                    int t_idx[4] = {i, j, c, o};
                    sum += in->atPadding(b_idx) * W_->at(t_idx);
                  }
                }
              }
              if(b_ != NULL) {
                sum += b_->at(0,0,0,o);
              }
              out->at(b, y_top, x_top, o) = sum;
            }
          }
        } else if (padding==VALID) {
          for(int x = kernel_width/2, x_top = 0; x_top < out->GetDims()[2]; x += stride, x_top += 1) {
            for(int y = kernel_height/2, y_top = 0; y_top < out->GetDims()[1]; y += stride, y_top += 1) {
              // batch idx b, output layer o, pixel (x, y)
              Dtype sum = 0.0;
              for(int c = 0; c < in_channels; c++) {
                for(int i = 0; i < kernel_height; i++) {
                  for(int j = 0; j < kernel_width; j++) {
                    // (n, hei, wid, channel),   // (hei, wid, input, output)
                    int b_idx[4] = {b, y+i-int(kernel_height/2), x+j-int(kernel_width/2), c};
                    int t_idx[4] = {i, j, c, o};
                    sum += in->atPadding(b_idx) * W_->at(t_idx);
                  }
                }
              }
              if(b_ != NULL) {
                sum += b_->at(0,0,0,o);
              } 
              out->at(b, y_top, x_top, o) = sum;
            }
          }
        }
      }
    } 
  } 


}





#endif // COMPUTATIONS_CUH_
