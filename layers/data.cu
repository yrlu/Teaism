
#ifndef CONV2D_LAYER_CUH_
#define CONV2D_LAYER_CUH_

#include <assert.h>
#include <stdio.h>
#include <string>
#include <sstream>
#include <iostream>
#include "basics/layer.hpp"
#include "basics/tensor.cu"
#include "utils/bitmap_image.hpp"
#include "basics/session.hpp"

// TODO: implement CUDA kernel for backward()

#define BLOCKDIM 32


template <class Dtype>
class Data: public Layer<Dtype> {
public:
  Data(unsigned batch_size, char img_list_path[]):
       batch_size(batch_size), begin_(0) {
    std::string img_path;
    size_t lab;

    std::ifstream file(img_list_path);
    std::string tmp;
    while(std::getline(file, tmp)) {
      std::istringstream iss(tmp);
      iss >> img_path;
      img_list.push_back(img_path);
      iss >> lab;
      lab_list.push_back(lab);
    }
    num_data_ = lab_list.size();
    assert(num_data_ > 0);
    assert(num_data_ >= batch_size);
    bitmap_image img(img_list[0]);
    img_h = img.height();
    img_w = img.width();
  }

  ~Data() {}

  void Forward(Tensor<Dtype>* bottom, Tensor<Dtype>* top) {}
  std::vector<Tensor<Dtype>* >& Forward(const std::vector<Tensor<Dtype> *> &bottom) {}
  std::vector<Tensor<Dtype>* > Forward();

  __host__ Tensor<Dtype>* FetchBatchDataCPU();

  // void Backward(Tensor& bottom, Tensor& top, Tensor& gradient) {}

  const size_t batch_size;
  std::vector<std::string> img_list;
  std::vector<size_t> lab_list;

  size_t img_h;
  size_t img_w;

private:
  size_t begin_;
  size_t end_;
  size_t num_data_;
//  size_t img_c;
};

template <class Dtype>
std::vector<Tensor<Dtype>* > Data<Dtype>::Forward() {
  end_ = begin_ + batch_size;
  if (end_ > num_data_) {
    begin_ = 0;
    end_ = begin_ + batch_size;
  }

  Tensor<Dtype>* top_t;
  size_t dims[4] = {batch_size, img_h, img_w, 3};
  top_t = FetchBatchDataCPU();
  
  if (Session::GetSession()->gpu) {
    Tensor<Dtype>* top_t_gpu = Tensor<Dtype>::TensorCPUtoGPU(top_t);
    delete top_t;
    top_t = top_t_gpu;
  }

  std::vector<Tensor<Dtype>* > top;
  top.push_back(top_t);
  return top;
}

template <class Dtype>
__host__ Tensor<Dtype>* Data<Dtype>::FetchBatchDataCPU() {
  size_t dims[4] = {batch_size, img_h, img_w, 3};
  Tensor<Dtype>* top_t = Tensor<Dtype>::CreateTensorCPU(dims);

  bitmap_image* img;
  for (size_t i = begin_; i < end_; ++i) {
    img = new bitmap_image(img_list[i]);
    for (size_t y = 0; y < img_h; ++y) {
      for (size_t x = 0; x < img_w; ++x) {
        top_t->at(i,y,x,0) = (Dtype) img->red_channel(x,y);
        top_t->at(i,y,x,1) = (Dtype) img->green_channel(x,y);
        top_t->at(i,y,x,2) = (Dtype) img->blue_channel(x,y);
      }
    }
    delete img;
  }
  return top_t;
}

#endif  // CONV2D_LAYER_CUH_
