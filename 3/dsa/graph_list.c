// Undirected graph

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct Node {
  int val;
  struct Node *next;
} Node;

typedef struct {
  int v;
  Node *vertices;
} Graph;

typedef struct {
  int *queue;
  int f, r, cap;
} Queue;

Graph *createGraph(int);
void addEdge(int, int, Graph *);
void bfs(Graph *, int, int *);
void dfs(Graph *, int, int *);

Node *newNode(int val);
void display(Graph *);
Queue *newQueue(int);
void enqueue(Queue *, int);
int dequeue(Queue *);

int main() {
  Graph *graph = createGraph(5);
  addEdge(1, 4, graph);
  addEdge(2, 3, graph);
  addEdge(2, 4, graph);
  addEdge(3, 4, graph);
  // display(graph);

  int *visitedNodes = (int *)calloc(graph->v, sizeof(int));
  int bfsStart = 2;

  printf("BFS with start node: %d => ", bfsStart);
  dfs(graph, bfsStart, visitedNodes);
  printf("\n");
}

Graph *createGraph(int vertices) {
  Graph *graph = (Graph *)malloc(sizeof(Graph));
  graph->v = vertices;
  graph->vertices = (Node *)malloc(vertices * sizeof(Node));

  for (int i = 0; i < vertices; i++) {
    graph->vertices[i].val = i;
    graph->vertices[i].next = NULL;
  }

  return graph;
}

void addEdge(int src, int dest, Graph *graph) {
  Node *node = newNode(dest);
  node->next = graph->vertices[src].next;
  graph->vertices[src].next = node;
  // printf("\nadded edge from %d,%p to %d,%p", src, graph->vertices[src].next,
  // dest, node->next);

  Node *nodeRev = newNode(src);
  nodeRev->next = graph->vertices[dest].next;
  graph->vertices[dest].next = nodeRev;
  // printf("\nadded edge from %d,%p to %d,%p\n", dest,
  // graph->vertices[dest].next,
  //        src, nodeRev->next);
}
void bfs(Graph *graph, int start, int *visited) {}

void dfs(Graph *graph, int start, int *visited) {
  if (visited[start] == 1) {
    return;
  }
  visited[start] = 1;
  Node node = graph->vertices[start];
  printf("%d ", node.val);
  Node *nextNode = node.next;
  while (nextNode != NULL) {
    if (visited[nextNode->val] != 1) {
      dfs(graph, nextNode->val, visited);
    }
    nextNode = nextNode->next;
  }
}

Node *newNode(int val) {
  Node *node = (Node *)malloc(sizeof(Node));
  node->val = val;
  node->next = NULL;

  return node;
}

void display(Graph *graph) {
  for (int i = 0; i < graph->v; i++) {
    printf("[%d]", graph->vertices[i].val);
    Node *head = graph->vertices[i].next;
    while (head != NULL) {
      printf(" -> %d ", head->val);
      head = head->next;
    }
    printf("\n");
  }
}

Queue *newStack(int cap) {
  Queue *queue = (Queue *)malloc(sizeof(Queue));
  queue->queue = (int *)malloc(cap * sizeof(int));
  queue->f = -1;
  queue->r = 0;
  queue->cap = cap;

  return queue;
}

void enqueue(Queue *queue, int node) {
  if (queue->f == -1) {
    queue->f++;
  }
  if (queue->r ==) {
    queue->queue[queue->r++] = node;
  }
}

int dequeue(Queue *queue) { return 0; }
