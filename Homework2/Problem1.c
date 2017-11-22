#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

#define MAP_SIZE 5000000
#define ARRAY_SIZE 6

#define MPI_TAG_INSERT 0
#define MPI_TAG_ERASE 1
#define MPI_TAG_COMPLETE 2

unsigned long combined(unsigned long first, unsigned long second) {
  return (first << 32) | second;
}

int map_index(unsigned long value) {
  return (value * 2654435761) % MAP_SIZE;
}

long combine_into_store(unsigned long old_value, unsigned long new_value, int size) {
  long *array = (long *)malloc(sizeof(long) * size);
  array[0] = (long)old_value;
  array[1] = (long)new_value;
  array[2] = -1;
  return -((long)array);
}

bool find_and_erase_value_in_store(long store, unsigned long value) {
  long *array = (long *)(-store);
  for (int i = 0; array[i] != -1; i++) {
    if (array[i] == (long)value) {
      array[i] = 0;
      return true;
    }
  }
  return false;
}

void insert_value_into_store(long store, unsigned long value) {
  long *array = (long *)(-store);
  for (int i = 0; true; i++) {
    if (array[i] == 0) {
      array[i] = (long)value;
      break;
    } else if (array[i] == -1) {
      array[i] = (long)value;
      array[i + 1] = -1;
      break;
    }
  }
}

long reduce_store_to_value_if_possible(long store) {
  long value = 0;
  long *array = (long *)(-store);
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

bool find_and_erase_value_in_map(long *map, int index, unsigned long value) {
  long store = map[index];
  if (store > 0) {
    if (store == (long)value) {
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

void insert_value_into_map(long *map, int index, unsigned long value, int max_array_size) {
  long store = map[index];
  if (store == 0) {
    map[index] = (long)value;
  } else if (store > 0) {
    map[index] = combine_into_store(store, value, max_array_size);
  } else if (store < 0) {
    insert_value_into_store(store, value);
    map[index] = reduce_store_to_value_if_possible(store);
  }
}

int other_node_recippar(int *edges, int nrow, int number_of_nodes, int current_node_index) {
  int count = 0;
  long *map = calloc(MAP_SIZE, sizeof(long));
  int partition = (nrow / (number_of_nodes - 1) + 1) * 4;
  for (int i = partition * (current_node_index - 1); i < partition * current_node_index && i < nrow * 2; i += 2) {
    unsigned long second_and_first = combined(edges[i + 1], edges[i]);
    if (find_and_erase_value_in_map(map, map_index(second_and_first), second_and_first)) {
      count += 1;
      MPI_Send(&second_and_first, 1, MPI_UNSIGNED_LONG, 0, MPI_TAG_ERASE, MPI_COMM_WORLD);
    } else {
      unsigned long first_and_second = combined(edges[i], edges[i + 1]);
      insert_value_into_map(map, map_index(first_and_second), first_and_second, ARRAY_SIZE);
      MPI_Send(&second_and_first, 1, MPI_UNSIGNED_LONG, 0, MPI_TAG_INSERT, MPI_COMM_WORLD);
    }
  }
  MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 0, MPI_TAG_COMPLETE, MPI_COMM_WORLD);
  return -1;
}

int first_node_recippar(int *edges, int number_of_nodes) {
  int count = 0;
  long *map = calloc(MAP_SIZE, sizeof(long));
  bool *finished_nodes = calloc(number_of_nodes, sizeof(bool));
  int number_of_finished_nodes = 0;
  while (number_of_finished_nodes != number_of_nodes - 1) {
    for (int i = 1; i < number_of_nodes; i++) {
      if (finished_nodes[i]) {
        continue;
      }
      MPI_Status status;
      unsigned long received;
      MPI_Recv(&received, 1, MPI_UNSIGNED_LONG, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      if (status.MPI_TAG == MPI_TAG_COMPLETE) {  // received complete action
        finished_nodes[i] = true;
        number_of_finished_nodes += 1;
        count += (int)received;
      } else if (status.MPI_TAG == MPI_TAG_INSERT) { // received insert action
        if (find_and_erase_value_in_map(map, map_index(received), received)) {
          count += 1;
        } else {
          unsigned long first_and_second = combined(received & 0xffffffff, received >> 32);
          insert_value_into_map(map, map_index(first_and_second), first_and_second, ARRAY_SIZE);
        }
      } else { // received erase action
        find_and_erase_value_in_map(map, map_index(received), received);
      }
    }
  }
  return count;
}

int recippar(int *edges, int nrow) {
  int number_of_nodes, current_node_index;
  MPI_Comm_size(MPI_COMM_WORLD, &number_of_nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &current_node_index);
  if (current_node_index == 0) { // start all nodes that aren't the first node
    return first_node_recippar(edges, number_of_nodes);
  } else { // start the last node
    return other_node_recippar(edges, nrow, number_of_nodes, current_node_index);
  }
}
