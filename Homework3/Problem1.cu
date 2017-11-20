#include <stdio.h>
#include <cuda.h>
#include <cublas.h>

__device__ float array_mean(float *x, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return sum / n;
}

__global__ void block_burst(float *x, int n, int k, long *meanIdx, float * meanData) {
  int start = blockIdx.x * blockDim.x * blockDim.y;
  int length = threadIdx.x + k;
  meanData[blockIdx.x * blockDim.x * blockDim.y + threadIdx.x] = array_mean(x + start, length);
  meanIdx[blockIdx.x * blockDim.x * blockDim.y + threadIdx.x] = (x + start) << 32 | length;
}

void maxburst(float *x, int n, int k, int *startend, float *bigmax) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float *device_x;
  int maxIdx = 0;
  cudaMalloc((void **)&device_x, sizeof(float) * n);
  cudaMemcpy(device_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);

  long* meanIdx;
  cudaMalloc((void **)&meanIdx, sizeof(long) * n);//need to change size change

  float *meanData;
  cudaMalloc((void **)&meanData, sizeof(float) * n);// size

  dim3 dimGrid(n - k + 1, 1); //
  dim3 dimBlock(n, 1, 1); //

  block_burst<<<dimGrid, dimBlock, (n - k + 1) * n>>>(device_x, n, k, meanIdx, meanData);

  cudaThreadSynchronize();

  cublasIsamax(handle, n, meanData, 0, &maxIdx);

  long index = meanIdx[maxIdx];
  startend[1] = index & 0xffffffff;
  startend[0] = index >> 32;
  *bigmax = meanData[maxIdx];
  cudaFree(meanIdx);
  cudaFree(meanData);
  cudaFree(device_x);
}

int main(int argc, char const *argv[]) {
  float x[7] = {1, 10, 3, 4, 3, 8, 1};
  int n = 7;
  int k = 3;
  int startend[2] = {0};
  float bigmax = 0;
  maxburst(x, n, k, startend, &bigmax);
  printf("%f (%d %d)\n", bigmax, startend[0], startend[1]);
  return 0;
}