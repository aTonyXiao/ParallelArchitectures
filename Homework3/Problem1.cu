#include <cuda.h>

__device__ float arraysum(float *x, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return sum;
}

__global__ void deviceburst(float *x, int n, int k, float *bigmaxs, int *startend) {
  int partition = (n - k + 1) / (blockDim.x * gridDim.x) + 1;
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
      left += length - k + 1;
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
  int blockDimX = 256 / ((k - 1) / 16000 + 1);
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

#include <stdio.h>
#include <sys/time.h>

int main() {
  int n = 450000;
  int k = 4000;
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
