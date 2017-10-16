#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>

#define NROW 1768149

#define ARRAY_SIZE 600

static inline double duration(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec) / 1000000.0;
}

static inline long combine_into_store(int old_value, int new_value, int size) {
  int *array = (int *)malloc(sizeof(int) * size);
  array[0] = old_value;
  array[1] = new_value;
  array[2] = -1;
  return -((long)array);
}

static inline bool find_and_erase_value_in_store(long store, int value) {
  int *array = (int *)(-store);
  int size = 0;
  for (int i = 0; array[i] != -1; i++) {
    size ++;
    if (array[i] == value) {
      array[i] = 0;
      return true;
    }
  }
  return false;
}

static inline void insert_value_into_store(long store, int value) {
  int *array = (int *)(-store);
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

static inline long reduce_store_to_value_if_possible(long store) {
  int value = 0;
  int *array = (int *)(-store);
  for (int i = 0; array[i] != -1; i++) {
    if (array[i] > 0) {
      if (value > 0) { // array has more than one value
        return store;
      } else {
        value = array[i];
      }
    }
  }
  return value;
}

static inline void insert_value_into_map(long *map, int value, int index, int max_array_size) {
  long store = map[index];
  if (store == 0) {
    map[index] = value;
  } else if (store > 0) {
    map[index] = combine_into_store((int)store, value, max_array_size);
  } else if (store < 0) {
    insert_value_into_store(store, value);
    map[index] = reduce_store_to_value_if_possible(store);
  }
}

int recippar(int *edges, int nrow) {
  int count = 0;
  long *map = calloc(INT_MAX, sizeof(long));
  #pragma omp for
  for (int i = 0; i < nrow * 2; i += 2) {
    int first = edges[i], second = edges[i + 1];
    long store = map[second];
    if (store == first) { // found a value
      map[second] = 0;
      count += 1;
    } else if (store >= 0) { // found nothing
      insert_value_into_map(map, second, first, ARRAY_SIZE);
    } else if (store < 0) { // found an array
      if (find_and_erase_value_in_store(store, first)) {
        count += 1;
      } else {
        insert_value_into_map(map, second, first, ARRAY_SIZE);
      }
    }
  }
  return count;
}

int main() {
  FILE *twitter_combined = fopen("twitter_combined.txt", "r");
  int *edges = (int *)malloc(sizeof(int) * NROW * 2);
  for (int i = 0; i < NROW * 2; i++) {
    fscanf(twitter_combined, "%d", &edges[i]);
  }
  struct timeval start, end;
  gettimeofday(&start, NULL);
  int count = recippar(edges, NROW);
  gettimeofday(&end, NULL);
  printf("Output: %d (%lfs)\n", count, duration(start, end));
  return 0;
}
