#include <stdio.h>

typedef struct {
  int *Graph;
  int numNodes;
} Graph;

Graph newGraph(int);

int main() {
  int numNode = 10;
  Graph graph = newGraph(10);
}
