
#include "layers/data.cu"
#include "layers/softmax.cu"
#include "layers/cross_entropy_loss.cu"
#include <vector>
#include <assert.h>

void test_get_top_dims() {
  size_t bs = 2;

  Data<float> data_layer(bs, "/home/jyh/github/Teaism/tmp/test/img_list.txt");
  
  std::vector<size_t*> data_bottoms_dims;
  data_bottoms_dims.push_back(new size_t[4]);
  
  std::vector<size_t*> data_tops_dims;
  data_tops_dims.push_back(new size_t[4]);
  data_tops_dims.push_back(new size_t[4]);

  data_layer.GetTopsDims(data_bottoms_dims, data_tops_dims);

  printf("Data Layer: \n");
  for (auto top_dims : data_tops_dims) {
    printf("(%d, %d, %d, %d)\n", top_dims[0], top_dims[1], top_dims[2], top_dims[3]);
  }


  Softmax<float> softmax_layer;

  std::vector<size_t*> softmax_bottoms_dims;
  softmax_bottoms_dims.push_back(data_tops_dims[0]);

  std::vector<size_t*> softmax_tops_dims;
  softmax_tops_dims.push_back(new size_t[4]);

  softmax_layer.GetTopsDims(softmax_bottoms_dims, softmax_tops_dims);

  printf("Softmax Layer: \n");
  for (auto top_dims : softmax_tops_dims) {
    printf("(%d, %d, %d, %d)\n", top_dims[0], top_dims[1], top_dims[2], top_dims[3]);
  }


  CrossEntropyLoss<float> cross_entropy_loss_layer;

  std::vector<size_t*> cel_bottoms_dims;
  size_t cel_bottom_dims[4] = {data_tops_dims[0][0], 1, 1, data_tops_dims[0][3]};
  cel_bottoms_dims.push_back(data_tops_dims[0]);
  cel_bottoms_dims.push_back(data_tops_dims[1]);

  std::vector<size_t*> cel_tops_dims;
  cel_tops_dims.push_back(new size_t[4]);

  cross_entropy_loss_layer.GetTopsDims(cel_bottoms_dims, cel_tops_dims);

  printf("Cross Entropy Loss Layer: \n");
  for (auto top_dims : cel_tops_dims) {
    printf("(%d, %d, %d, %d)\n", top_dims[0], top_dims[1], top_dims[2], top_dims[3]);
  }


}


int main() {
  test_get_top_dims();
}
