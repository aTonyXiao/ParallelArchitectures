#include <stdio.h>
#include <cuda.h>


__device__ float array_mean(float *x, int n) {
  // printf("x[0]: %f\n", x[0]);
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return sum / n;
}


__global__ void block_burst(float *x, int n, int k, int *result) {
  cublasHandle_t handle
  int start = blockIdx.x;
  int length = threadIdx.x + k;
  result[2] = 0;
  int padding = 0;
  extern __shared__ float currentmax[];
  currentmax[threadIdx.x] = array_mean(x + start, length);
  int Maxidx;
  // long resultIdx = mall
  printf("currentmax: %f\n", currentmax);
  __syncthreads();

  cublasStatus_t cublasIsamax(handle, n, currentmax, 0, &Maxidx);


  // if (currentmax > result[2]) {
  //   result[0] = start;
  //   result[1] = start + length - 1;
  //   result[2] = currentmax;
  //   printf("result[2]: %f\n", result[2]);
  // }
}


void maxburst(float *x, int n, int k, int *startend, float *bigmax) {
  float *device_x;
  cudaMalloc((void **)&device_x, sizeof(float) * n);
  cudaMemcpy(device_x, x, sizeof(float) * n, cudaMemcpyHostToDevice);
  int *device_result; // extern __shared__ int device_result[];
  cudaMalloc((void **)&device_result, sizeof(int) * 3);
  dim3 dimGrid(n - k + 1, 1);
  dim3 dimBlock(n, 1, 1);
  block_burst<<<dimGrid, dimBlock, (n - k + 1) * n>>>(device_x, n, k, device_result);
  cudaMemcpy(startend, device_result, sizeof(int) * 2, cudaMemcpyDeviceToHost);
  cudaMemcpy(bigmax, device_result + 2, sizeof(int) * 1, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();
  cudaFree(device_x);
  cudaFree(device_result);
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
