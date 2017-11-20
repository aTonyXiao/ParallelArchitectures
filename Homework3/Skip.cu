// CSIF
// clear && /usr/local/cuda-8.0/bin/nvcc -Wno-deprecated-gpu-targets -g -G Skip.cu && a.out

#include <stdio.h>
#include <cuda.h>

__device__ float arraysum(float *x, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return sum;
}

__global__ void deviceburst(float *x, int n, int k, float *bigmaxs, int *startend) {
  // TODO: Better partition?
  int partition = n / (blockDim.x * gridDim.x) + 1;
  int me = blockIdx.x * gridDim.x + threadIdx.x;

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
  printf("range %d to %d maxmean: %f (from %d to %d)\n", left, left_limit, bigmaxs[me], startend[me * 2], startend[me * 2 + 1]);
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
  int gridDimX = 2; // 65536;
  int blockDimX = 2; // 512;
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

  float *bigmaxs = (float *)malloc(sizeof(float) * threads_count);
  cudaMemcpy(bigmaxs, device_bigmaxs, sizeof(float) * threads_count, cudaMemcpyDeviceToHost);
  cudaFree(device_bigmaxs);

  int *startends = (int *)malloc(sizeof(int) * threads_count * 2);
  cudaMemcpy(startends, device_startends, sizeof(int) * threads_count * 2, cudaMemcpyDeviceToHost);
  cudaFree(device_startends);

  int maxidx = arraymaxidx(bigmaxs, threads_count);
  bigmax[0] = bigmaxs[maxidx];
  startend[0] = startends[maxidx * 2];
  startend[1] = startends[maxidx * 2 + 1];
}

int main() {
  float input[] = {1, 2, 3, 4, 5, 4, 8, 9, 9};
  int startend[] = {0, 0};
  float bigmax = 0;
  maxburst(input, 9, 3, startend, &bigmax);
  printf ("%f (from %d to %d)\n", bigmax, startend[0], startend[1]);
}
