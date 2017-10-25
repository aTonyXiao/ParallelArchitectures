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

int numNodes, currentNode;
double startTime, endTime;

void Initial(int argc, char ** argv) //inital MPI
{
  int debugWait;
  debugWait = atoi(argv[1]);
  while(debugWait);
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numNodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &currentNode);
  if(currentNode == numNodes - 1) {
    startTime = MPI_Wtime();
  }
}


void Nodes(int *edges, int partition) //all nodes except the last one
{
  long *local_map = calloc(INT_MAX, sizeof(long));
  long local_count = findLocal(local_map, edges, partition);
  MPI_Send(&local_count, 1, MPI_INT, numNodes - 1, LOCAL_COUNT, MPI_COMM_WORLD);
}


void NodeEnd(int *edges, int partition)
{
  MPI_Status status;
  long *map = calloc(INT_MAX, sizeof(long));
  int count = 0;
  long received;

  while(1){
    int nodeArray[numNodes] = {0};

    for(int i = 0; i < numNodes - 1; i++) {
      if(nodeArray[i] != 0)
        continue;

      MPI_Recv(&received, 1, MPI_LONG, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      switch (status.MPI_TAG) {
        case LOCAL_COUNT: {
          nodeArray[i] ++;
          nodeArray[numNodes - 1] ++;
          count += (int)received;
          break;
        }
        case FOUND_VALUE: {
          map[((int)received)] = 0;
          break;
        }
        case INSERT_MAP: {
          int second = (int)(received / pow(2, 32));
          int first = (int)(received % pow(2, 32));
          insert_value_into_map(local_map, second, first, ARRAY_SIZE);
          break;
        }
        case ERASE_MAP: {
          int second = (int)(received / pow(2, 32));
          int first = (int)(received % pow(2, 32));
          long store = map[second];
          find_and_erase_value_in_store(store, first);
          break;
        }
        default: break;
      }
    }
    if(nodeArray[numNodes - 1] == numNodes)
      break;
  }

  for (int i = 1; i < INT_MAX; i++) { //search recippar in merged map
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
  printf("count : %d", count);

}

double duration(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}

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


long findLocal(long *local_map, int *edges, int partition) {
  long local_count = 0;
  long sum;
  for (int i = partition * currentNode / 2; i < (partition * currentNode + 1) / 2; i += 2) {
    int first = edges[i], second = edges[i + 1];
    long store = local_map[second];
    sum = second * pow(2, 32);
    sum += first;
    if (store == first) { // found a value
      sum = (long)second;
      MPI_Send(&sum, 1, MPI_LONG, numNodes - 1, FOUND_VALUE, MPI_COMM_WORLD);
      local_map[second] = 0;
      local_count += 1;
    } else if (store >= 0) { // found nothing
      insert_value_into_map(local_map, second, first, ARRAY_SIZE);
      MPI_Send(&sum, 1, MPI_LONG, numNodes - 1, INSERT_MAP, MPI_COMM_WORLD);
    } else if (store < 0) { // found an array
      if (find_and_erase_value_in_store(store, first)) {
        MPI_Send(&sum, 1, MPI_LONG, numNodes - 1, ERASE_MAP, MPI_COMM_WORLD);
        local_count += 1;
      } else {
        MPI_Send(&sum, 1, MPI_LONG, numNodes - 1, INSERT_MAP, MPI_COMM_WORLD);
        insert_value_into_map(local_map, second, first, ARRAY_SIZE);
      }
    }
  }
  return local_count;
}



int recippar(int *edges, int nrow) {
  int count = 0;


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
