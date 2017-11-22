#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

float arraySum(float *x, int k) {
  float sum = 0;
  for (int i = 0; i < k; i++)
    sum += x[i];

  return sum;
}

float maxburst(float *arr, int len, int k) {
  float sum = arraySum(arr, k);
  float maxMean = 0;
  float mean;
  int start = 0, length = k;

  do {
    float next = arr[start + length];
    mean = sum / length;

    if (next > mean) {
      if (next > arr[start]) {
        sum -= arr[start];
        sum += next;
        start++;
      }
      else { // next < arr[start], adding
        length++;
        sum += next;
      }
    }
    else { // lower than mean, jump
      if (maxMean < mean)
        maxMean = mean;

      start = start + length - k + 1;
      length = k;
      sum = arraySum(arr + start, length);
    }
  } while (start + length < len);

  mean = sum / length;
  return mean > maxMean ? mean : maxMean;
}

// int main() {
//   float arr[] = {1,2,3,4,5,4,8,9,9};
//   printf ("%f\n", maxburst(arr, 9, 3));
// }
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
  maxburst(x, n, k, &bigmax);
  struct timeval end;
  gettimeofday(&end, NULL);
  float duration = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
  printf("%f (from %d to %d) (%fms)\n", bigmax, startend[0], startend[1], duration);
}
