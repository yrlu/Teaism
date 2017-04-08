
#include<stdlib>

class Tensor
{
public:
  Tensor(vector<size_t> dim) {
    allocate_data(dim);
  };
   ~Tensor() {};

//  __device__ void set_data();
  __device__ float* get_data() {
    return data_array_;
  }

  __device__ unsigned get_idx(vector<unsigned> idx) {
    unsigned out_idx = 0;
    for (unsigned i = idx.size(); i >= 0; --i)
      out_idx = out_idx*dim[i] + idx[i];
    return out_idx;
  }

  __device__ float get_data_at_idx(vector<unsigned> idx) {
    return data_array_[get_idx(idx)];
  }

private:
  float* data_array_ // on GPUs 

  void allocate_data(vector<size_t> dim) {

    // Allocate data_array_ on GPUs
    // To-do: 3DcudaMalloc for image processing
    unsigned total_bytes = std::accumulate(dim.begin(), dim.end(), 1, std::multiplies<size_t>);
    cudaExtent extent = make_cudaExtent(total_bytes, 1, 1);
    cudaStatus = cudaMalloc(data_array_, extent);
    checkCudaErrors(cudaStatus);

  }

};

