#ifndef LAYER_HPP_
#define LAYER_HPP_

#include "tensor.cu"
#include <vector>

template <class Dtype>
class Layer {
public:
  virtual void Forward(const std::vector<Tensor<Dtype>*> &bottoms, const std::vector<Tensor<Dtype>*> &tops) = 0;
  
<<<<<<< HEAD
  virtual void Backward(const std::vector<Tensor<Dtype>*> &tops,
                const std::vector<Tensor<Dtype>*> &tops_diff,
                const std::vector<Tensor<Dtype>*> &bottoms,
                const std::vector<Tensor<Dtype>*> &bottoms_diff) = 0;

=======
  /*virtual void Backward(const std::vector<Tensor<Dtype>*> &tops,
                        const std::vector<Tensor<Dtype>*> &tops_diff,
                        const std::vector<Tensor<Dtype>*> &bottoms,
                        const std::vector<Tensor<Dtype>*> &bottoms_diff) = 0;
  */
>>>>>>> c32464aec4b22ad21e63ddc44b375133e0b13f58
  virtual void GetTopsDims(const std::vector<size_t*> &bottoms_dims, const std::vector<size_t*> &tops_dims) = 0;
};


#endif  // LAYER_HPP_
