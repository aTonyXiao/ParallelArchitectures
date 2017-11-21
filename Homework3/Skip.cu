#include <stdio.h>
#include <cuda.h>
#include "cublas_v2.h"

__device__ float arraysum(float *x, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return sum;
}

__global__ void deviceburst(float *x, int n, int k, float *bigmaxs, int *startend) {
  int partition = n / (blockDim.x * gridDim.x) + 1;
  int me = blockIdx.x * blockDim.x + threadIdx.x;

  int left = me * partition;
  int left_limit = left + partition;
  int length = k;

  float sum = arraysum(x + left, length);
  float mean = sum / length;

  startend[me * 2] = left;
  startend[me * 2 + 1] = left + length - 1;
  bigmaxs[me] = mean;

  while (left + length < n && left < left_limit) {
    float next = x[left + length];
    if (next > mean) {
      if (next > x[left]) {
        sum = sum + next - x[left];
        left += 1;
      } else {
        sum = sum + next;
        length += 1;
      }
    } else {
      left = left + length - k + 1;
      length = k;
      sum = arraysum(x + left, length);
    }
    mean = sum / length;
    if (mean > bigmaxs[me]) {
      startend[me * 2] = left;
      startend[me * 2 + 1] = left + length - 1;
      bigmaxs[me] = mean;
    }
  }
}

int arraymaxidx(float *x, int n) {
  float max = x[0];
  int maxidx = 0;
  for (int i = 1; i < n; i++) {
    if (x[i] > max) {
      max = x[i];
      maxidx = i;
    }
  }
  return maxidx;
}

void maxburst(float *x, int n, int k, int *startend, float *bigmax) {
  int gridDimX = 128;
  int blockDimX = 256;
  int threads_count = gridDimX * blockDimX;

  float *device_x;
  cudaMalloc((void **)&device_x, sizeof(float) * n);
  cudaMemcpy(device_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);

  float *device_bigmaxs;
  cudaMalloc((void **)&device_bigmaxs, sizeof(float) * threads_count);

  int *device_startends;
  cudaMalloc((void **)&device_startends, sizeof(int) * threads_count * 2);

  dim3 dimGrid(gridDimX, 1);
  dim3 dimBlock(blockDimX, 1, 1);
  deviceburst<<<dimGrid, dimBlock>>>(device_x, n, k, device_bigmaxs, device_startends);
  cudaThreadSynchronize();

  cudaFree(device_x);

  cublasHandle_t handle;
  cublasCreate(&handle);
  int maxidx;
  cublasIsamax(handle, threads_count, device_bigmaxs, 1, &maxidx);
  maxidx -= 1;
  cudaMemcpy(bigmax, device_bigmaxs + maxidx, sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(startend, device_startends + maxidx * 2, sizeof(int) * 2, cudaMemcpyDeviceToHost);
  cudaFree(device_bigmaxs);
  cudaFree(device_startends);
  cublasDestroy(handle);
}

// ----
// Testing
//
// CSIF
// clear && /usr/local/cuda-8.0/bin/nvcc -Wno-deprecated-gpu-targets -g -G Skip.cu && a.out

#include <sys/time.h> // TODO: Remove before submit

int main() {
  int n = 500000;
  int k = 200;
  float *x = (float *)malloc(sizeof(float) * n);
  srand(0);
  for (int i = 0; i < n; i++) {
    x[i] = (float)rand() / (float)(RAND_MAX / 100.0);
  }
  int startend[] = {0, 0};
  float bigmax = 0;
  struct timeval start;
  gettimeofday(&start, NULL);
  maxburst(x, n, k, startend, &bigmax);
  struct timeval end;
  gettimeofday(&end, NULL);
  float duration = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
  printf("%f (from %d to %d) (%fms)\n", bigmax, startend[0], startend[1], duration);
}
