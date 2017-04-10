
#include "tensor.cuh"

class Packet
{
public:
  Packet(Tensor* data, Tensor* gradient):
    data_(data), gradient_(gradient) {};
  ~Packet() {};

  __device__ void set_data();
  __device__ Tensor* get_data{ return data_ }
  __device__ Tensor* get_gradient{ return gradient_ }

private:
  Tensor* data_;
  Tensor* gradient_;

};
