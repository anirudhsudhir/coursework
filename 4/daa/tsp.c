#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  int cost;
  int *path;
} CombinationResult;

CombinationResult generate_combination(int pathData[], int nextPathDataIndex,
                                       int n, int **graph);

int main() {
  int n;
  scanf("%d", &n);
  int **graph;

  graph = malloc(n * sizeof(int *));
  for (int i = 0; i < n; i++) {
    graph[i] = malloc(n * sizeof(int));
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      scanf("%d", &graph[i][j]);
    }
  }

  int pathData[n + 1];
  pathData[0] = 0;
  CombinationResult res = generate_combination(pathData, 1, n, graph);

  if (res.cost <= 0) {
    printf("path does not exist\n");
    return 0;
  }

  printf("Min Cost = %d\n", res.cost);
  printf("Path: ");
  for (int i = 0; i < n + 1; i++) {
    printf("%d ", res.path[i]);
  }
  printf("\n");
}

// generating all possible combinations
CombinationResult generate_combination(int pathData[], int nextPathDataIndex,
                                       int n, int **graph) {
  if (nextPathDataIndex == n) {
    // calculate cost of the path
    int cost = 0;
    for (int i = 0; i < n - 1; i++)
      cost += graph[pathData[i]][pathData[i + 1]];

    // adding path and cost from final to inital node
    pathData[n] = pathData[0];
    cost += graph[pathData[n - 1]][pathData[n]];

    CombinationResult res = {cost, pathData};
    return res;
  }

  CombinationResult curRes;
  curRes.cost = INT_MAX;

  // all possible combinations for the current index
  for (int i = 0; i < n; i++) {

    // checking if node i already exists in pathData (already visited)
    int flag = 0;
    for (int j = 0; j < nextPathDataIndex; j++) {
      if (pathData[j] == i) {
        flag = 1;
        break;
      }
    }

    if (flag == 1)
      continue;

    pathData[nextPathDataIndex] = i;
    CombinationResult res =
        generate_combination(pathData, nextPathDataIndex + 1, n, graph);

    if (res.cost < curRes.cost) {
      curRes.cost = res.cost;
      curRes.path = res.path;
    }
  }

  return curRes;
}
