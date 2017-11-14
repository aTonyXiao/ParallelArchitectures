#include <stdio.h>

float array_mean(float *x, int n) {
  float sum = 0;
  for (int i = 0; i < n; i++) {
    sum += x[i];
  }
  return sum / n;
}

void maxburst(float *x, int n, int k, int *startend, float *bigmax) {
  *bigmax = 0;
  for (int start = 0; start <= n - k; start++) {
    for (int length = k; length <= n - start; length++) {
      float currentmax = array_mean(x + start, length);
      if (currentmax > *bigmax) {
        *bigmax = currentmax;
        startend[0] = start;
        startend[1] = start + length - 1;
      }
    }
  }
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
