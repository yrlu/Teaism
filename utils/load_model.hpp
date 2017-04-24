#ifndef LOAD_MODEL_HPP_
#define LOAD_MODEL_HPP_

#include <string>
#include <sstream>
#include <iostream>
#include "basics/tensor.cu"

template <class Dtype>
void load_to_conv(Tensor<Dtype>* t, std::ifstream &file) {
  std::string tmp_string;
  std::string tmp_token;
  Dtype tmp;
  const size_t* dims = t->GetDims();

  printf("Loading conv: (%d, %d, %d, %d): \n", dims[0],dims[1],dims[2],dims[3]);
  std::getline(file, tmp_string);
  std::istringstream iss(tmp_string);
  for (int o = 0; o < dims[3]; ++o) {
    for (int i = 0; i < dims[2]; ++i) {
      for (int w = 0; w < dims[0]; ++w) {
        for (int h = 0; h < dims[1]; ++h) {
          iss >> tmp_token;
          tmp = std::stof(tmp_token);
          t->at(w,h,i,o) = tmp;
        }
      }
    }
  }
}

template <class Dtype>
void load_to_fc(Tensor<Dtype>* t, std::ifstream &file) {
  std::string tmp_string;
  std::string tmp_token;
  Dtype tmp;
  const size_t* dims = t->GetDims();

  printf("Loading fc: (%d, %d, %d, %d): \n", 1,1,dims[2],dims[3]);
  std::getline(file, tmp_string);
  std::istringstream iss(tmp_string);
  for (int o = 0; o < dims[2]; ++o) {
    for (int i = 0; i < dims[3]; ++i) {
      iss >> tmp_token;
      tmp = std::stof(tmp_token);
      t->at(0,0,o,i) = tmp;
    }
  }
}

template <class Dtype>
void load_to_bias(Tensor<Dtype>* t, std::ifstream &file) {
  std::string tmp_string;
  std::string tmp_token;
  Dtype tmp;
  const size_t* dims = t->GetDims();

  printf("Loading bias: (%d, %d, %d, %d): ", 1,1,1,dims[3]);
  std::getline(file, tmp_string);
  std::istringstream iss(tmp_string);
  for (int o = 0; o < dims[3]; ++o) {
    iss >> tmp_token;
    tmp = std::stof(tmp_token);
    t->at(0,0,0,o) = tmp;
  }
}

#endif  // LOAD_MODEL_HPP_
