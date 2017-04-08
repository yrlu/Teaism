

Class ConvLayer
{
public:
  ConvLayer(unsigned filter_height, unsigned filter_width, 
    unsigned in_channels, unsigned out_channels, unsigned stride):
    filter_height_(filter_height), filter_width_(filter_width), 
    in_channels_(in_channels), out_channels_(out_channels), stride_(stride) {}

  ~ConvLayer() {}

  void forward(Packet* bottom, Packet* top) {
    // startTimer();
    //assert(Tensor.shape==);
    dim3 blocksInGrid(img.width() / BLOCKDIM, img.height() / BLOCKDIM);
    dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
    blurGlobalSharedPixels << <blocksInGrid, threadsPerBlock >> > (d_src, d_dst, d_gaussian);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    checkCudaErrors(cudaStatus);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    checkCudaErrors(cudaStatus);

    printf("Kernel time:  %3.1f ms \n", stopTimer());
  }

  void backword(Packet* bottom, Packet* top) {

  }

private:
  const unsigned filter_height_;
  const unsigned filter_width_;
  const unsigned in_channels_;
  const unsigned out_channels_;
  const unsigned stride_;
};

