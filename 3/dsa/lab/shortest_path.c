#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  int **g;
  int n, c;
} Graph;

typedef struct {
  int *q;
  int cap, f, r;
} Queue;

Graph *newGraph(int n, int c) {
  Graph *graph = (Graph *)malloc(sizeof(Graph));
  graph->g = (int **)calloc(n, sizeof(int *));
  graph->n = n;
  graph->c = c;

  for (int i = 0; i < n; i++) {
    graph->g[i] = (int *)calloc(n, sizeof(int));
  }

  return graph;
}

Queue *newQueue(int cap) {
  Queue *q = (Queue *)calloc(1, sizeof(Queue));
  q->q = (int *)calloc(cap, sizeof(int));
  q->cap = cap;
  q->f = -1;
  q->r = 0;

  return q;
}

void enqueue(Queue *q, int ele) {
  if (q->f == -1) {
    q->f++;
  }
  if (q->r < q->cap) {
    q->q[q->r++] = ele;
    // printf("enqueued %d\n",ele);
  } else {
    printf("QUEUE OVERFLOW!");
  }
}

int dequeue(Queue *q) {
  if (q->f == -1) {
    printf("QUEUE UNDERFLOW!");
    return -1;
  }
  int ele = q->q[q->f++];
  // printf("dequeued %d\n",ele);
  if (q->f == q->r) {
    q->f = -1;
    q->r = 0;
  }
  return ele;
}

int isEmpty(Queue *q) { return (q->f == -1 && q->r == 0); }

void addEdge(Graph *graph, int srcEdge, int destEdge) {
  graph->g[srcEdge][destEdge] = 1;
  graph->g[destEdge][srcEdge] = 1;
}

void displayGraph(Graph *g) {
  for (int i = 0; i < g->n; i++) {
    printf("[%d] - ", i);
    for (int j = 0; j < g->n; j++) {
      printf(" %d ", g->g[i][j]);
    }
    printf("\n");
  }
}

void pathFinder(Graph *g, int src, int dest) {
  // printf("starting path find\n");
  Queue *q = newQueue(g->n);
  int *visited = (int *)calloc(g->n, sizeof(int));
  int *dist = (int *)calloc(g->n, sizeof(int));

  if (src == dest) {
    printf("True\n0");
    return;
  }

  if (g->g[src][dest] == 1) {
    printf("True\n1");
    return;
  }

  visited[src] = 1;
  enqueue(q, src);
  while (!isEmpty(q)) {
    int currentNode = dequeue(q);
    for (int i = 0; i < g->n; i++) {
      int nextNode = g->g[currentNode][i];
      if (nextNode == 1 && visited[i] != 1) {
        dist[i] = dist[currentNode] + 1;
        if (i == dest) {
          printf("True\n%d", dist[i]);
          return;
        }
        enqueue(q, i);
      }
    }
  }
  printf("False");
}

int main() {
  int n, c, src, dest;
  scanf("%d", &n);
  scanf("%d", &c);
  scanf("%d", &src);
  scanf("%d", &dest);

  Graph *graph = newGraph(n, c);

  for (int i = 0; i < c; i++) {
    int startEdge, endEdge;
    scanf("%d %d", &startEdge, &endEdge);
    addEdge(graph, startEdge, endEdge);
  }

  // displayGraph(graph);
  pathFinder(graph, src, dest);
  return 0;
}
