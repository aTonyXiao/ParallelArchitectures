#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <sys/time.h>

#define NROW 1768149

int min(int i0, int i1) {
  return i0 > i1 ? i1 : i0;
}

double duration(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_usec - t0.tv_usec) / 1000.0;
}

int recippar(int *edges, int nrow) {
  int count = 0;
  int *found = calloc(INT_MAX, sizeof(int));
  for (int i = 0; i < nrow * 2; i += 2) {
    int first = edges[i];
    int second = edges[i + 1];
    long found_first = found[second];
    if (found_first == first) { // found a value
      found[second] = 0;
      // count += 1;
    } else if (found_first == 0) { // found nothing
      found[first] = second;
    } else {
      count += 1;
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
  struct timeval start;
  gettimeofday(&start, NULL);
  int count = recippar(edges, NROW);
  struct timeval end;
  gettimeofday(&end, NULL);
  printf("Count: %d\nDuration: %lf\n", count, duration(start, end));
  return 0;
}
