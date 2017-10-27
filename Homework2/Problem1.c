#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

#define NROW 1768149
#define FOUND_VALUE 0
#define INSERT_MAP 1
#define ERASE_MAP 2
#define LOCAL_COUNT 3

#define ARRAY_SIZE 1000

int number_of_nodes, current_node_index;

long combine_into_store(int old_value, int new_value, int size) {
  int *array = (int *)malloc(sizeof(int) * size);
  array[0] = old_value;
  array[1] = new_value;
  array[2] = -1;
  return -((long)array);
}

bool find_and_erase_value_in_store(long store, int value) {
  int *array = (int *)(-store);
  for (int i = 0; array[i] != -1; i++) {
    if (array[i] == value) {
      array[i] = 0;
      return true;
    }
  }
  return false;
}

void insert_value_into_store(long store, int value) {
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

long reduce_store_to_value_if_possible(long store) {
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
  free(array);
  return value;
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
    if (find_and_erase_value_in_store(store, value)) {
      map[index] = reduce_store_to_value_if_possible(store);
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
    map[index] = combine_into_store((int)store, value, max_array_size);
  } else if (store < 0) {
    insert_value_into_store(store, value);
    map[index] = reduce_store_to_value_if_possible(store);
  }
}

void insert_store_into_map(long *map, long store, int index, int max_array_size) {
  int *array = (int *)(-store);
  for (int i = 0; array[i] != -1; i++) {
    int value = array[i];
    if (value > 0) {
      insert_value_into_map(map, value, index, max_array_size);
    }
  }
}

void initialize(int argc, char ** argv) //inital MPI
{
  int debugWait;
  debugWait = atoi(argv[1]);
  while(debugWait);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &number_of_nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &current_node_index);
}

unsigned long node_recippar(long *local_map, int *edges, int partition, int nrow) {
  unsigned long local_count = 0;
  for (int i = partition * current_node_index; i < partition * (current_node_index + 1) && i < nrow * 2; i += 2) {
    unsigned long first = edges[i], second = edges[i + 1];
    long store = local_map[second];
    unsigned long to_send = (second << 32) | first;
    if (store == first) { // found a value
      local_map[second] = 0;
      local_count += 1;
      MPI_Send(&to_send, 1, MPI_UNSIGNED_LONG, number_of_nodes - 1, FOUND_VALUE, MPI_COMM_WORLD);
    } else if (store >= 0) { // found nothing
      to_send |= ((unsigned long)1) << 31;
      insert_value_into_map(local_map, second, first, ARRAY_SIZE);
      MPI_Send(&to_send, 1, MPI_UNSIGNED_LONG, number_of_nodes - 1, FOUND_VALUE, MPI_COMM_WORLD);
    } else if (store < 0) { // found an array
      if (find_and_erase_value_in_store(store, first)) {
        local_count += 1;
      } else {
        to_send |= ((unsigned long)1) << 31;
        insert_value_into_map(local_map, second, first, ARRAY_SIZE);
      }
      MPI_Send(&to_send, 1, MPI_UNSIGNED_LONG, number_of_nodes - 1, FOUND_VALUE, MPI_COMM_WORLD);
    }
  }
  return local_count;
}

void start_node(int *edges, int partition, int nrow) // all nodes except the last one
{
  long *local_map = calloc(INT_MAX, sizeof(long));
  unsigned long local_count = node_recippar(local_map, edges, partition, nrow);
  MPI_Send(&local_count, 1, MPI_UNSIGNED_LONG, number_of_nodes - 1, LOCAL_COUNT, MPI_COMM_WORLD);
}


unsigned long start_last_node(int *edges, int partition)
{
  unsigned long count = 0;
  long *map = calloc(INT_MAX, sizeof(long));
  bool *finished_nodes = calloc(number_of_nodes, sizeof(bool));
  int number_of_finished_nodes = 0;
  while (true) {
    for (int i = 0; i < number_of_nodes - 1; i++) {
      if (finished_nodes[i]) {
        continue;
      }

      MPI_Status status;
      unsigned long received;
      MPI_Recv(&received, 1, MPI_UNSIGNED_LONG, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      if (status.MPI_TAG == LOCAL_COUNT) {
        finished_nodes[i] = true;
        number_of_finished_nodes += 1;
        count += received;
        continue;
      }

      unsigned long first = received & 0xffffffff;
      unsigned long second = received >> 32;
      unsigned short insert_or_erase = (first & 0x80000000) >> 31;
      second &= 0x7fffffff;
      first &= 0x7fffffff;
      long store = map[second];

      switch (insert_or_erase) {
        case 0: { // ERASE
          find_and_erase_value_in_map(map, first, second);
          break;
        }
        case 1: { // INSERT
          insert_value_into_map(map, second, first, ARRAY_SIZE);
          break;
        }
        default: {
          break;
        }
      }
    }
    if (number_of_finished_nodes == number_of_nodes - 1) {
      break;
    }
  }

  printf("Count (Before Merge): %lu\n", count);
  for (int i = 1; i < INT_MAX / 3; i++) { //search recippar in merged map
    long store = map[i];
    if (store > 0 && store != i) { // found a value
      if (find_and_erase_value_in_map(map, i, store)) {
        count += 1;
      }
    } else if (store < 0) { // found an array
      int *values = (int *)(-store);
      for (int index = 0; values[index] != -1; index++) {
        if (values[index] > 0 && values[index] != i) {
          if (find_and_erase_value_in_map(map, i, values[index])) {
            count += 1;
          }
        }
      }
    }
  }
  return count;
}

int recippar(int *edges, int nrow) {
  int partition = ((int)ceil((double)nrow / (double)(number_of_nodes - 1) / 2)) * 4;
  if (current_node_index != number_of_nodes - 1) { // start all nodes that aren't the last node
    start_node(edges, partition, nrow);
    return -1; // return -1 to indicate that the node will not calculate the end result
  } else { // start the last node
    return (int)start_last_node(edges, partition);
  }
}

int main(int argc, char** argv) {
  FILE *twitter_combined = fopen("twitter_combined.txt", "r");
  int *edges = (int *)malloc(sizeof(int) * NROW * 2);
  for (int i = 0; i < NROW * 2; i++) {
    fscanf(twitter_combined, "%d", &edges[i]);
  }
  initialize(argc, argv);
  double start_time, end_time;
  MPI_Barrier(MPI_COMM_WORLD);
  start_time = MPI_Wtime();
  int count = recippar(edges, NROW);
  struct timeval end;
  end_time = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  if (count == -1) {
    printf("Node %d finished.\n", current_node_index);
  } else {
    printf("Count (%lfs): %d\n", end_time - start_time, count);
  }
  return 0;
}
