#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>

#define NROW 1768149

#define ARRAY_SIZE 30

int min(int i0, int i1) {
  return i0 > i1 ? i1 : i0;
}

double duration(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}

int recippar(int *edges, int nrow) {
  int count = 0;
  long *found = calloc(INT_MAX, sizeof(long));
  omp_set_num_threads(4);
  #pragma omp parallel
  {
    int thread_num = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int local_count = 0;
    long *local_found = calloc(INT_MAX, sizeof(long));
    for (int i = thread_num; i < nrow * 2; i += nth * 2) {
      int first = edges[i];
      int second = edges[i + 1];
      long found_first = local_found[second];
      if (found_first == first) { // found a value
        local_found[second] = 0;
        local_count += 1;
      } else if (found_first == 0) { // found nothing
        local_found[first] = second;
      } else if (found_first < 0) { // found an array
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
            array[insert_location + 1] = -1;
            break;
          }
        }
        if (insert_location != -1) {
          array[insert_location] = second;
        }
      } else { // found a value that is not the same
        int *array = (int *)malloc(sizeof(int) * ARRAY_SIZE);
        array[0] = found_first;
        array[1] = second;
        array[2] = -1;
        local_found[first] = -((long)array);
      }
    }
    for (long i = 0; i < INT_MAX; i++) {
      if (local_found[i] != 0) {
        #pragma omp critical
        {
          if (found[i] > 0) { // found[i] has an integer
            int *array = (int *)malloc(sizeof(int) * ARRAY_SIZE * nth);
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
                array[insert_location] = local_array[i];
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
    for (long i = 0; i < INT_MAX; i++) {
      long j = found[i];
      if (j > 0) { // found a value
        long first = found[j];
        if (first == i) { // found a value
          count += 1;
          found[i] = 0;
          found[j] = 0;
        } else if (first >= 0) { // found nothing
          found[i] = 0;
        } else { // found an array
          int *inside_array = (int *)(-first);
          for (int actual_j = 0; true; actual_j++) {
            if (first == inside_array[actual_j]) {
              count += 1;
              found[i] = 0;
              inside_array[actual_j] = 0;
              break;
            } else if (inside_array[actual_j] == -1) {
              found[i] = 0;
              break;
            }
          }
        }
      } else if (j < 0) { // found an array
        int *array = (int *)(-j);
        for (int i_index = 0; array[i_index] != -1; i_index++) {
          int actual_j = array[i_index];
          long first = found[actual_j];
          if (first == i) { // found a value
            count += 1;
            array[i_index] = 0;
            found[actual_j] = 0;
          } else if (first >= 0) { // found nothing
            array[i_index] = 0;
          } else { // found an array
            int *inside_array = (int *)(-first);
            for (int first_index = 0; true; first_index++) {
              if (first == inside_array[first_index]) {
                count += 1;
                array[i_index] = 0;
                inside_array[first_index] = 0;
                break;
              } else if (inside_array[first_index] == -1) {
                array[i_index] = 0;
                break;
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
  int *edges = (int *)malloc(sizeof(int) * NROW * 2);
  for (int i = 0; i < NROW * 2; i++) {
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
