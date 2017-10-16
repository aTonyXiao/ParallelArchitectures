#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>

#define NROW 1768149

#define ARRAY_SIZE 20

int min(int i0, int i1) {
  return i0 > i1 ? i1 : i0;
}

double duration(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}

int *combine_into_array(int old_value, int new_value, int size) {
  int *array = (int *)malloc(sizeof(int) * size);
  array[0] = old_value;
  array[1] = new_value;
  array[2] = -1;
  return array;
}

bool find_and_erase_value_in_array(int *array, int value) {
  for (int i = 0; array[i] != -1; i++) {
    if (array[i] == value) {
      array[i] = 0;
      return true;
    }
  }
  return false;
}

void insert_value_into_array(int *array, int value) {
  for (int i = 0; true; i++) {
    if (array[i] == 0) {
      array[i] = value;
      break;
    } else if (array[i] == -1) {
      array[i] = value;
      array[i + 1] = -1;
      break;
    }
  }
}

int *reduce_array_to_value_if_possible(int *array) {
  int value = 0;
  for (int i = 0; array[i] != -1; i++) {
    if (array[i] > 0) {
      if (value > 0) { // array has more than one value
        return array;
      } else {
        value = array[i];
      }
    }
  }
  free(array); // TODO: check if can speed things up
  return (int *)((long)value);
}

bool find_and_erase_value_in_map(long *map, int value, int index) {
  long store = map[index];
  if (store > 0) {
    if (store == value) {
      map[index] = 0;
      return true;
    } else {
      return false;
    }
  } else if (store < 0) {
    int *array = (int *)(-store);
    if (find_and_erase_value_in_array(array, value)) {
      map[index] = (long)reduce_array_to_value_if_possible(array); // TODO: check if can speed things up
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

void insert_value_into_map(long *map, int value, int index, int max_array_size) {
  long store = map[index];
  if (store == 0) {
    map[index] = value;
  } else if (store > 0) {
    int *array = combine_into_array((int)store, value, max_array_size);
    map[index] = -((long)array);
  } else if (store < 0) {
    int *array = (int *)(-store);
    insert_value_into_array(array, value);
    map[index] = (long)reduce_array_to_value_if_possible(array);
  }
}

// TODO: check if rewrite this function can speed things up
void insert_values_into_map(long *map, int *values, int index, int max_array_size) {
  for (int i = 0; values[i] != -1; i++) {
    int value = values[i];
    if (value > 0) {
      insert_value_into_map(map, value, index, max_array_size);
    }
  }
}

int recippar(int *edges, int nrow) {
  int count = 0;
  long *map = calloc(INT_MAX, sizeof(long));
  omp_set_num_threads(4);
  #pragma omp parallel
  {
    int thread_num = omp_get_thread_num();
    int nth = omp_get_num_threads();
    int local_count = 0;
    long *local_map = calloc(INT_MAX, sizeof(long));
    for (int i = thread_num * 2; i < nrow * 2; i += nth * 2) {
      int first = edges[i], second = edges[i + 1];
      long store = local_map[second];
      if (store == first) { // found a value
        local_map[second] = 0;
        local_count += 1;
      } else if (store >= 0) { // found nothing
        insert_value_into_map(local_map, second, first, ARRAY_SIZE);
      } else if (store < 0) { // found an array
        int *values = (int *)(-store);
        if (find_and_erase_value_in_array(values, first)) {
          local_count += 1;
        } else {
          insert_value_into_map(local_map, second, first, ARRAY_SIZE);
        }
      }
    }
    #pragma omp critical
    {
      printf("Thread %d Count: %d\n", thread_num, local_count);
      count += local_count;
    }
    for (int i = 0; i < INT_MAX; i++) {
      long store = local_map[i];
      if (store > 0) { // found a value
        #pragma omp critical
        {
          insert_value_into_map(map, store, i, ARRAY_SIZE * nth);
        }
      } else if (store < 0) { // found an array
        #pragma omp critical
        {
          int *values = (int *)(-store);
          insert_values_into_map(map, values, i, ARRAY_SIZE * nth);
        }
      }
    }
  }
  #pragma omp barrier
  {
    printf("Count (Before Merge): %d\n", count);
    for (int first = 0; first < INT_MAX; first++) {
      long store = map[first];
      if (store > 0) { // found a value
        int second = (int)store;
        if (find_and_erase_value_in_map(map, first, second)) {
          count += 1;
        }
      } else if (store < 0) { // found an array
        int *values = (int *)(-store);
        for (int i = 0; values[i] != -1; i++) {
          int second = values[i];
          if (second > 0) {
            if (find_and_erase_value_in_map(map, first, second)) {
              count += 1;
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
