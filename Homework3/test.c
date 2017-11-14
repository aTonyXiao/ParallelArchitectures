#include <stdio.h>

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

int main() {
  float arr[] = {1,2,3,4,5,4,8,9,9};
  printf ("%f\n", maxburst(arr, 9, 3));
}
