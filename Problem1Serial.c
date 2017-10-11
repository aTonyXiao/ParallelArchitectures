#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/time.h>

#define NROW 100000000
#define MAXNUMBER 10

double duration(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}

int recippar(int **edges, int nrow) {
  int count = 0;
  int found[MAXNUMBER][MAXNUMBER] = {{0}};
  for (int i = 0; i < nrow; i++) {
    int first = edges[i][0];
    int second = edges[i][1];
    if (found[second][first] > 0) {
      found[second][first] -= 1;
      count += 1;
    } else {
      found[first][second] += 1;
    }
  }
  return count;
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
}
