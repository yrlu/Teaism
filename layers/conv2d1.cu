#include "layers/conv2d.cuh"



template <class Dtype>
void Conv2D<Dtype>::Forward(Tensor<Dtype> * bottom, Tensor<Dtype> * top) 


template class Conv2D<float>;
template class Conv2D<double>;
template class Conv2D<int>;