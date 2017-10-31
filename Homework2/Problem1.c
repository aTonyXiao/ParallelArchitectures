#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

#define MAP_SIZE 4000000

#define MPI_TAG_INSERT 0
#define MPI_TAG_ERASE 1
#define MPI_TAG_COMPLETE 2

unsigned long combined(unsigned long first, unsigned long second) {
  return (first << 32) | second;
}

int hash_value(unsigned long value) {
  return (value * 2654435761) % MAP_SIZE;
}

bool find_and_erase_value_in_map(long *map, unsigned long value) {
  int index = hash_value(value);
  while (map[index] != 0) {
    if (map[index] == (long)value) {
      map[index] = -1;
      return true;
    }
    index = (index * 2) % MAP_SIZE;
  }
  return false;
}

void insert_value_into_map(long *map, unsigned long value) {
  int index = hash_value(value);
  while (map[index] > 0) {
    index = (index * 2) % MAP_SIZE;
  }
  map[index] = value;
}

int other_node_recippar(int *edges, int nrow, int number_of_nodes, int current_node_index) {
  int count = 0;
  long *map = calloc(MAP_SIZE, sizeof(long));
  int partition = (nrow / (number_of_nodes - 1) + 1) * 4;
  for (int i = partition * (current_node_index - 1); i < partition * current_node_index && i < nrow * 2; i += 2) {
    unsigned long second_and_first = combined(edges[i + 1], edges[i]);
    if (find_and_erase_value_in_map(map, second_and_first)) {
      count += 1;
      MPI_Send(&second_and_first, 1, MPI_UNSIGNED_LONG, 0, MPI_TAG_ERASE, MPI_COMM_WORLD);
    } else {
      unsigned long first_and_second = combined(edges[i], edges[i + 1]);
      insert_value_into_map(map, first_and_second);
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
        if (find_and_erase_value_in_map(map, received)) {
          count += 1;
        } else {
          unsigned long first_and_second = combined(received & 0xffffffff, received >> 32);
          insert_value_into_map(map, first_and_second);
        }
      } else { // received erase action
        find_and_erase_value_in_map(map, received);
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

int main(int argc, char** argv) {
  int NROW = 1768149;
  FILE *twitter_combined = fopen("twitter_combined.txt", "r");
  int *edges = (int *)malloc(sizeof(int) * NROW * 2);
  for (int i = 0; i < NROW * 2; i++) {
    fscanf(twitter_combined, "%d", &edges[i]);
  }
  MPI_Init(&argc, &argv);
  double start_time, end_time;
  MPI_Barrier(MPI_COMM_WORLD);
  start_time = MPI_Wtime();
  int count = recippar(edges, NROW);
  struct timeval end;
  end_time = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  if (count != -1) {
    printf("Count (%lfs): %d\n", end_time - start_time, count);
  }
  return 0;
}
