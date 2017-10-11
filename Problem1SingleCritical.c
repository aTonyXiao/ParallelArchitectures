#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>
#include <omp.h>

#define NROW 100000000
#define MAXNUMBER 10

int min(int i0, int i1) {
  return i0 > i1 ? i1 : i0;
}

double duration(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}

int recippar(int **edges, int nrow) {
  int count = 0;
  int found[MAXNUMBER][MAXNUMBER] = {{0}};
  #pragma omp parallel
  {
    int local_found[MAXNUMBER][MAXNUMBER] = {{0}};
    int local_count = 0;
    int me = omp_get_thread_num();
    int nth = omp_get_num_threads();
    for (int i = me; i < nrow; i += nth) {
      int first = edges[i][0];
      int second = edges[i][1];
      if (local_found[second][first] > 0) {
        local_found[second][first] -= 1;
        local_count += 1;
      } else {
        local_found[first][second] += 1;
      }
    }
    #pragma omp critical
    {
      for (int i = 0; i < MAXNUMBER; i++) {
        for (int j = 0; j < MAXNUMBER; j++) {
          found[i][j] += local_found[i][j];
        }
      }
      count += local_count;
    }
  }
  #pragma omp barrier
  {
    for (int i = 0; i < MAXNUMBER; i++) {
      for (int j = 0; j < MAXNUMBER; j++) {
        if (j > i) {
          continue;
        } else if (i == j) {
          count += found[i][j] / 2;
        } else {
          count += min(found[i][j], found[j][i]);
        }
      }
    }
    return count;
  }
}

int main() {
  int **edges = (int **)malloc(sizeof(int *) * NROW);
  for (int i = 0; i < NROW; i++) {
    edges[i] = (int *)malloc(sizeof(int) * 2);
    edges[i][0] = rand() % MAXNUMBER;
    edges[i][1] = rand() % MAXNUMBER;
  }
  struct timeval start;
  gettimeofday(&start, NULL);
  int count = recippar(edges, NROW);
  struct timeval end;
  gettimeofday(&end, NULL);
  printf("Count: %d\nDuration: %lf\n", count, duration(start, end));
  return 0;
}
