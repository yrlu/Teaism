
#include <stdio.h>
#include <string>
#include <sstream>
#include <iostream>
#include "basics/tensor.cu"
#include "layers/conv2d.cu"
#include "layers/fc.cu"


void test_load_model() {

  std::string model_path = "models/alexnet/model.txt";

  std::ifstream file(model_path);
  std::string tmp_string;
  float tmp_float;
  while(std::getline(file, tmp_string)) {
//    std::istringstream iss(tmp);
//    iss >> tmp_string;
    std::cout << tmp_string << std::endl;
    tmp_float = std::stof(tmp_string);
    std::cout << tmp_float << std::endl;
  }

}


int main() {
  test_load_model();
}
