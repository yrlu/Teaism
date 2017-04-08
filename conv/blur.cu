
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

// c/o: http://www.partow.net/programming/bitmap/index.html
#include "bitmap_image.hpp"

#include <stdio.h>

const double PI = 3.14159265358979323846;

const int BLOCKDIM = 32;

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

/** Compute a Gaussian blur of src image and place into dst. Use only global memory. */
__global__ void blurGlobal(cudaPitchedPtr src, cudaPitchedPtr dst, float* gaussian) {

  /* Swap x and y in threadIdx */
	int x = (blockDim.x * blockIdx.x) + threadIdx.x;
	int y = (blockDim.y * blockIdx.y) + threadIdx.y;

	float r = 0.0, g = 0.0, b = 0.0;

	for (int ky = 0; ky < FILTER_SIZE; ky++) {
		for (int kx = 0; kx < FILTER_SIZE; kx++) {
			// this replicates border pixels
			int i = index(clamp(x + kx - FILTER_HALFSIZE, src.xsize / 4),
				clamp(y + ky - FILTER_HALFSIZE, src.ysize), src);
			unsigned int pixel = ((int*)src.ptr)[i];
			// convolute each channel separately
			const float k = gaussian[(ky * FILTER_SIZE) + kx];
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


/** Compute a Gaussian blur of src image and place into dst. Gaussian kernel in shared memory. */
__global__ void blurGlobalSharedGaussian(cudaPitchedPtr src, cudaPitchedPtr dst, float* gaussian) {

  /* Initialize Gaussian kernel in shared memory. */
  __shared__ float sharedGaussian[FILTER_SIZE][FILTER_SIZE];
  if (threadIdx.x < FILTER_SIZE && threadIdx.y < FILTER_SIZE)
    sharedGaussian[threadIdx.x][threadIdx.y] = gaussian[(threadIdx.y)*FILTER_SIZE + threadIdx.x];
  __syncthreads();

  int x = (blockDim.x * blockIdx.x) + threadIdx.x;
  int y = (blockDim.y * blockIdx.y) + threadIdx.y;

  float r = 0.0, g = 0.0, b = 0.0;

  for (int ky = 0; ky < FILTER_SIZE; ky++) {
    for (int kx = 0; kx < FILTER_SIZE; kx++) {
      // this replicates border pixels
      int i = index(clamp(x + kx - FILTER_HALFSIZE, src.xsize / 4),
        clamp(y + ky - FILTER_HALFSIZE, src.ysize), src);
      unsigned int pixel = ((int*)src.ptr)[i];
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

// used by startTimer() and stopTimer()
cudaEvent_t start, stop;

void startTimer() {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

/** Return elapsed time (in ms) since startTime() was called */
float stopTimer() {
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	return time;
}

struct pixel_t {
	unsigned char __unused, red, green, blue;
};

int main() {

	const char* INPUT_BMP_PATH = "./test/steel_wool_large.bmp";
	const char* OUTPUT_REFERENCE_BMP_PATH = "./test/steel_wool_large_reference_output.bmp";
	const char* OUTPUT_BMP_PATH = "./test/out.bmp";

	// LOAD IMAGE FROM FILE
	bitmap_image img(INPUT_BMP_PATH);

	if (!img) {
		printf("Error - Failed to open: %s \r\b", INPUT_BMP_PATH);
		return 1;
	}

	// ensure that image dimensions are a multiple of the block size
	if (img.height() % BLOCKDIM != 0) {
		printf("ERROR: image height (%d) must be a multiple of the block size (%d)\n", img.height(), BLOCKDIM);
		return 1;
	}
	if (img.width() % BLOCKDIM != 0) {
		printf("ERROR: image width (%d) must be a multiple of the block size (%d)\n", img.width(), BLOCKDIM);
		return 1;
	}

	const int IMG_WIDTH_BYTES = img.width() * 4;

	// we store pixels in a 32-bit int of the form 0x00bbggrr (8 bits for each of the blue, green, and red channels)
	pixel_t* h_buf = new pixel_t[img.width() * img.height()];

	// fill up h_buf
	for (unsigned int y = 0; y < img.height(); y++) {
		for (unsigned int x = 0; x < img.width(); x++) {
			pixel_t* pixel = &h_buf[(y * img.width()) + x];
			img.get_pixel(x, y, pixel->red, pixel->green, pixel->blue);
		}
	}

	cudaError_t cudaStatus;

	// use 48KB for shared memory, and 16KB for L1D$
	cudaStatus = cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	checkCudaErrors(cudaStatus);

	// ensure kernel timeout is disabled
	int kernelTimeout;
	cudaStatus = cudaDeviceGetAttribute(&kernelTimeout, cudaDevAttrKernelExecTimeout, 0/*device*/);
	checkCudaErrors(cudaStatus);
	if (kernelTimeout != 0) {
		printf("WARNING: kernel timeout is enabled! %d \r\n", kernelTimeout);
	}

	// COPY IMAGE BUFFERS AND FILTER TO DEVICE
	startTimer();
	cudaExtent extent = make_cudaExtent(IMG_WIDTH_BYTES, img.height(), 1);
	cudaPitchedPtr d_src, d_dst;
	cudaStatus = cudaMalloc3D(&d_src, extent);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaMemcpy2D(d_src.ptr, d_src.pitch,
		h_buf, IMG_WIDTH_BYTES, IMG_WIDTH_BYTES, img.height(),
		cudaMemcpyHostToDevice);
	checkCudaErrors(cudaStatus);

	cudaStatus = cudaMalloc3D(&d_dst, extent);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaMemset2D(d_dst.ptr, d_dst.pitch, 0, IMG_WIDTH_BYTES, img.height());
	checkCudaErrors(cudaStatus);

	float* d_gaussian;
	setupGaussian(&d_gaussian);

	printf("Copy to device:  %3.1f ms \n", stopTimer());

	// LAUNCH KERNEL

	for (int i = 0; i < 5; i++) {
		startTimer();
		dim3 blocksInGrid(img.width() / BLOCKDIM, img.height() / BLOCKDIM);
		dim3 threadsPerBlock(BLOCKDIM, BLOCKDIM);
		//blurGlobal << <blocksInGrid, threadsPerBlock >> > (d_src, d_dst, d_gaussian);
		//blurGlobalSharedGaussian << <blocksInGrid, threadsPerBlock >> > (d_src, d_dst, d_gaussian);
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

	// COPY  OUTPUT IMAGE BACK TO HOST
	startTimer();
	cudaStatus = cudaMemcpy2D(h_buf, IMG_WIDTH_BYTES,
		d_dst.ptr, d_dst.pitch, IMG_WIDTH_BYTES, d_dst.ysize,
		cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaStatus);

	printf("Copy from device:  %3.1f ms \n", stopTimer());

	// WRITE OUT UPDATED IMAGE
	for (unsigned int y = 0; y < img.height(); y++) {
		for (unsigned int x = 0; x < img.width(); x++) {
			const pixel_t* p = &h_buf[(y * img.width()) + x];
			img.set_pixel(x, y, p->red, p->green, p->blue);
		}
	}
	img.save_image(OUTPUT_BMP_PATH);

	// VALIDATION

	bool validated = true;
	bitmap_image ref(OUTPUT_REFERENCE_BMP_PATH);

	if (img.height() != ref.height()) {
		fprintf(stderr, "Image height should be %u but was %u \r\n", ref.height(), img.height());
		validated = false;
	}
	if (img.width() != ref.width()) {
		fprintf(stderr, "Image width should be %u but was %u \r\n", ref.width(), img.width());
		validated = false;
	}
	unsigned int differingPixels = 0;
	double squareDiffSum = 0;
	for (unsigned int y = 0; y < ref.height(); y++) {
		for (unsigned int x = 0; x < ref.width(); x++) {
			rgb_t refPixel, imgPixel;
			ref.get_pixel(x, y, refPixel);
			img.get_pixel(x, y, imgPixel);
			if (refPixel.red != imgPixel.red ||
				refPixel.green != imgPixel.green ||
				refPixel.blue != imgPixel.blue) {
				differingPixels++;
				
				// compute square difference
				unsigned int redDiff = refPixel.red - imgPixel.red;
				unsigned int greenDiff = refPixel.green - imgPixel.green;
				unsigned int blueDiff = refPixel.blue - imgPixel.blue;
				squareDiffSum += (redDiff * redDiff) + (greenDiff * greenDiff) + (blueDiff * blueDiff);
			}
		}
	}
	if (0 != differingPixels) {
		fprintf(stderr, "Found %u pixels that differ from the reference image \r\n", differingPixels);
		double rmsd = sqrt(squareDiffSum / (ref.height() * ref.width()));
		fprintf(stderr, "RMSD of pixel rgb values is %3.5f \r\n", rmsd);
		validated = false;
	}

	if (validated) {
		printf("Validation passed :-) \r\n");
	}


	// CLEANUP

	cudaStatus = cudaFree(d_src.ptr);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaFree(d_dst.ptr);
	checkCudaErrors(cudaStatus);
	cudaStatus = cudaFree(d_gaussian);
	checkCudaErrors(cudaStatus);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	checkCudaErrors(cudaStatus);

	delete[] h_buf;

	return 0;
}
