
// Note: this must be an odd number
__device__ const int FILTER_SIZE = 15;
__device__ const int FILTER_HALFSIZE = FILTER_SIZE >> 1;

__device__ const int BLUE_MASK = 0x00ff0000;
__device__ const int GREEN_MASK = 0x0000ff00;
__device__ const int RED_MASK = 0x000000ff;

/** compute index into an int* for the pixel (x,y) in the given 2D pitched allocation */
__device__ int index(int x, int y, const cudaPitchedPtr& cpp) {
	// divide by 4 because each pixel is 4B and cpp.pitch is in bytes, but we need to return an index
	return (y * (cpp.pitch / 4)) + x;
}

/** Clamp the given value to the interval [0,bound) */
__device__ int clamp(int value, int bound) {
	if (value < 0) {
		return 0;
	}
	if (value < bound) {
		return value;
	}
	return bound - 1;
}

/** Compute a Gaussian blur of src image and place into dst. Gaussian kernel and pixels in shared memory. */
__global__ void blurGlobalSharedPixels(cudaPitchedPtr src, cudaPitchedPtr dst, float* gaussian) {

  /* Initialize Gaussian kernel in shared memory */
  __shared__ float sharedGaussian[FILTER_SIZE][FILTER_SIZE];
  if (threadIdx.x < FILTER_SIZE && threadIdx.y < FILTER_SIZE)
    sharedGaussian[threadIdx.x][threadIdx.y] = gaussian[(threadIdx.y)*FILTER_SIZE + threadIdx.x];

  /* Initialize pixel buffer in shared memory */
  int x = (blockDim.x * blockIdx.x) + threadIdx.x;
  int y = (blockDim.y * blockIdx.y) + threadIdx.y;

  __shared__ unsigned int sharedPixels[BLOCKDIM+2*FILTER_HALFSIZE][BLOCKDIM+2*FILTER_HALFSIZE];
  int i = index(clamp(x, src.xsize/4), clamp(y, src.ysize), src);
  sharedPixels[threadIdx.x+FILTER_HALFSIZE][threadIdx.y+FILTER_HALFSIZE] = ((int*)src.ptr)[i];

  if (threadIdx.x < FILTER_HALFSIZE && threadIdx.y < FILTER_HALFSIZE) {
    i = index(clamp(x-FILTER_HALFSIZE, src.xsize/4), clamp(y-FILTER_HALFSIZE, src.ysize), src);
    sharedPixels[threadIdx.x][threadIdx.y] = ((int*)src.ptr)[i];
  }
  if (threadIdx.x < FILTER_HALFSIZE) {
    i = index(clamp(x-FILTER_HALFSIZE, src.xsize/4), clamp(y, src.ysize), src);
    sharedPixels[threadIdx.x][threadIdx.y+FILTER_HALFSIZE] = ((int*)src.ptr)[i];
  }
  if (threadIdx.y < FILTER_HALFSIZE) {
    i = index(clamp(x, src.xsize/4), clamp(y-FILTER_HALFSIZE, src.ysize), src);
    sharedPixels[threadIdx.x+FILTER_HALFSIZE][threadIdx.y] = ((int*)src.ptr)[i];
  }
  if (threadIdx.x >= BLOCKDIM-FILTER_HALFSIZE && threadIdx.y >= BLOCKDIM-FILTER_HALFSIZE) {
    i = index(clamp(x+FILTER_HALFSIZE, src.xsize/4), clamp(y+FILTER_HALFSIZE, src.ysize), src);
    sharedPixels[threadIdx.x+2*FILTER_HALFSIZE][threadIdx.y+2*FILTER_HALFSIZE] = ((int*)src.ptr)[i];
  }
  if (threadIdx.x >= BLOCKDIM-FILTER_HALFSIZE) {
    i = index(clamp(x+FILTER_HALFSIZE, src.xsize/4), clamp(y, src.ysize), src);
    sharedPixels[threadIdx.x+2*FILTER_HALFSIZE][threadIdx.y+FILTER_HALFSIZE] = ((int*)src.ptr)[i];
  }
  if (threadIdx.y >= BLOCKDIM-FILTER_HALFSIZE) {
    i = index(clamp(x, src.xsize/4), clamp(y+FILTER_HALFSIZE, src.ysize), src);
    sharedPixels[threadIdx.x+FILTER_HALFSIZE][threadIdx.y+2*FILTER_HALFSIZE] = ((int*)src.ptr)[i];
  }
  if (threadIdx.x < FILTER_HALFSIZE && threadIdx.y >= BLOCKDIM-FILTER_HALFSIZE) {
    i = index(clamp(x-FILTER_HALFSIZE, src.xsize/4), clamp(y+FILTER_HALFSIZE, src.ysize), src);
    sharedPixels[threadIdx.x][threadIdx.y+2*FILTER_HALFSIZE] = ((int*)src.ptr)[i];
  }
  if (threadIdx.x >= BLOCKDIM-FILTER_HALFSIZE && threadIdx.y < FILTER_HALFSIZE) {
    i = index(clamp(x+FILTER_HALFSIZE, src.xsize/4), clamp(y-FILTER_HALFSIZE, src.ysize), src);
    sharedPixels[threadIdx.x+2*FILTER_HALFSIZE][threadIdx.y] = ((int*)src.ptr)[i];
  }

  __syncthreads();

  float r = 0.0, g = 0.0, b = 0.0;

  for (int ky = 0; ky < FILTER_SIZE; ky++) {
    for (int kx = 0; kx < FILTER_SIZE; kx++) {
      unsigned int pixel = sharedPixels[threadIdx.x+kx][threadIdx.y+ky];
      // convolute each channel separately
      const float k = sharedGaussian[kx][ky];
      b += (float)((pixel & BLUE_MASK) >> 16) * k;
      g += (float)((pixel & GREEN_MASK) >> 8) * k;
      r += (float)((pixel & RED_MASK)) * k;
    }
  }
  // Re-assemble destination pixel
  unsigned int dpixel = 0x00000000
    | ((((int)b) << 16) & BLUE_MASK)
    | ((((int)g) << 8) & GREEN_MASK)
    | (((int)r) & RED_MASK);
  ((int*)dst.ptr)[index(x, y, dst)] = dpixel;
}


void setupGaussian(float** d_gaussian) {
	// calculate gaussian blur filter
	float gaussian[FILTER_SIZE][FILTER_SIZE];
	double sigma = 5.0;
	double mean = FILTER_SIZE / 2;
	for (int x = 0; x < FILTER_SIZE; ++x) {
		for (int y = 0; y < FILTER_SIZE; ++y) {
			double g = exp(-0.5 * (pow((x - mean) / sigma, 2.0) + pow((y - mean) / sigma, 2.0))) / (2 * PI * sigma * sigma);
			gaussian[y][x] = (float)g;
		}
	}
	// normalize the filter
	float sum = 0.0;
	for (int x = 0; x < FILTER_SIZE; ++x) {
		for (int y = 0; y < FILTER_SIZE; ++y) {
			sum += gaussian[y][x];
		}
	}
	for (int x = 0; x < FILTER_SIZE; ++x) {
		for (int y = 0; y < FILTER_SIZE; ++y) {
			gaussian[y][x] /= sum;
		}
	}

	// copy gaussian to device memory
	cudaError_t cudaStatus = cudaMalloc(d_gaussian, FILTER_SIZE * FILTER_SIZE * sizeof(float));
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMemcpy(*d_gaussian, &gaussian[0], FILTER_SIZE * FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);
}

