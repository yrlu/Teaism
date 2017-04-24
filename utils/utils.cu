#ifndef UTILS_LAYER_CUH_
#define UTILS_LAYER_CUH_

__device__ inline unsigned GetIdx(const size_t * dims, const int* idx) {
  unsigned out_idx = 0;
  for(int i = 0; i < 3; i++) {
    out_idx = out_idx*dims[i] + idx[i];
  }
  return out_idx;
}

__device__ inline unsigned GetIdx(const size_t * dims, const int idx0, const int idx1, const int idx2) {
  int idx[3];
  idx[0] = idx0; idx[1] = idx1; idx[2] = idx2;
  return GetIdx(dims, idx);
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

// show memory usage of GPU
void show_mem(cudaError_t cuda_status) {
  size_t free_byte;
  size_t total_byte;
  cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
  if(cudaSuccess!=cuda_status) {
  	printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
  	return;
  }
  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;

  printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
  used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}




#endif // UTILS_LAYER_CUH_