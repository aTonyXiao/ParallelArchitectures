#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>

#define NROW 4841532

int min(int i0, int i1) {
  return i0 > i1 ? i1 : i0;
}

double duration(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}

int recippar(int *edges, int nrow) {
  int count = 0;
  long found[INT_MAX] = {0};
  #pragma omp parallel
  {
    int thread_num = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int local_count = 0;
    long local_found[INT_MAX] = {0};
    // int *local_found = calloc(INT_MAX, sizeof(int));
    for (int i = thread_num; i < nrow; i += nth * 2) {
      int first = edges[i];
      int second = edges[i + 1];
      long found_first = local_found[second];
      if (found_first == first) {
        local_found[second] = 0;
        local_count += 1;
      } else if (found_first == 0) {
        local_found[first] = second;
      } else if (found_first < 0) {
        int *array = (int *)(-found_first);
        int insert_location = -1;
        for (int i = 0; true; i++) {
          if (array[i] == first) {
            array[i] = 0;
            insert_location = -1;
            local_count += 1;
            break;
          } else if (array[i] == 0 && insert_location == -1) {
            insert_location = i;
          } else if (array[i] == -1) {
            insert_location = i;
            break;
          }
        }
        if (insert_location != -1) {
          array[insert_location] = first;
        }
      } else {
        int *array = (int *)malloc(sizeof(int) * 5);
        array[0] = found_first;
        array[1] = first;
        array[2] = -1;
        local_found[first] = -((long)array);
      }
    }
    for (int i = 0; i < INT_MAX; i++) {
      if (local_found[i] != 0) {
        #pragma omp critical
        {
          if (found[i] > 0) { // found[i] has an integer
            int *array = (int *)malloc(sizeof(int) * 5 * nth);
            array[0] = found[i];
            if (local_found[i] > 0) { // local_found[i] has an integer
              array[1] = local_found[i];
              array[2] = -1;
            } else if (local_found[i] < 0) { // local_found[i] has an array of integers
              int *local_array = (int *)(-local_found[i]);
              for (int i = 0; true; i++) {
                array[i + 1] = local_array[i];
                if (local_array[i] == -1) {
                  break;
                }
              }
            }
            found[i] = -((long)array);
          } else if (found[i] == 0) { // found[i] has nothing
            found[i] = local_found[i];
          } else { // found[i] has an array of integers
            int *array = (int *)(-found[i]);
            int insert_location = 2;
            for (insert_location = 2; array[insert_location] != -1; insert_location++);
            if (local_found[i] > 0) { // local found has an integer
              array[insert_location] = local_found[i];
              array[insert_location + 1] = -1;
            } else if (local_found[i] < 0) { // local found has an array of integers
              int *local_array = (int *)(-local_found[i]);
              for (int i = 0; true; i++) {
                array[insert_location + i] = local_array[i];
                if (local_array[i] == -1) {
                  break;
                }
              }
            }
          }
        }
      }
    }
    #pragma omp critical
    {
      count += local_count;
    }
  }
  #pragma omp barrier
  {
    for (int i = 0; i < INT_MAX; i++) {
      long j = found[i];
      if (j > 0) { // found a value
        long first = found[j];
        if (first == i) { // found a value
          count += 1;
          found[i] = 0;
          found[j] = 0;
        } else if (first == 0) { // found nothing
          found[i] = 0;
        } else { // found an array
          int *array = (int *)(-first);
          for (int k = 0; array[k] != -1; k++) {
            if (first == array[k]) {
              count += 1;
              found[i] = 0;
              array[k] = 0;
            }
          }
        }
      } else if (j < 0) { // found an array
        int *array = (int *)(-j);
        for (int k = 0; array[k] != -1; k++) {
          int actual_j = array[k];
          long first = found[actual_j];
          if (first == i) { // found a value
            count += 1;
            array[k] = 0;
            found[actual_j] = 0;
          } else if (first == 0) { // found nothing
            array[k] = 0;
          } else { // found an array
            int *second_array = (int *)(-first);
            for (int m = 0; second_array[m] != -1; m++) {
              if (first == second_array[m]) {
                count += 1;
                array[k] = 0;
                second_array[m] = 0;
              }
            }
          }
        }
      }
    }
    return count;
  }
}

int main() {
  FILE *twitter_combined = fopen("twitter_combined.txt", "r");
  int *edges = (int *)malloc(sizeof(int) * NROW);
  for (int i = 0; i < NROW; i++) {
    fscanf(twitter_combined, "%d", &edges[i]);
  }
  struct timeval start;
  gettimeofday(&start, NULL);
  int count = recippar(edges, NROW);
  struct timeval end;
  gettimeofday(&end, NULL);
  printf("Count: %d\nDuration: %lf\n", count, duration(start, end));
  return 0;
}
