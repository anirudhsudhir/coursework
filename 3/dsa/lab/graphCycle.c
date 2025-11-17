#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
  int vertex;
  struct Node *next;
} Node;

typedef struct Graph {
  int numVertices;
  Node **adjLists;
  bool *visited;
} Graph;

// The createGraph function initializes a new graph with a specified number of
// vertices. It allocates memory for the adjacency list and visited array for
// each vertex setting each list entry to NULL and marking all vertices as
// unvisited
Graph *createGraph(int vertices);

// As the graph is undirected, this function should add an edge from both src to
// dest and dest to src
void addEdge(Graph *graph, int src, int dest);

// Returns true for cyclic graph, otherwise false
bool containsCycle(Graph *graph);

void display(Graph *g) {
  for (int i = 0; i < g->numVertices; i++) {
    printf("[%d]", i);
    Node *temp = g->adjLists[i]->next;
    while (temp != NULL) {
      printf(" -> %d", temp->vertex);
      temp = temp->next;
    }
    printf("\n");
  }
}

int main() {
  int n, m;
  scanf("%d %d", &n, &m);

  Graph *graph = createGraph(n);

  for (int i = 0; i < m; i++) {
    int u, v;
    scanf("%d %d", &u, &v);
    addEdge(graph, u, v);
  }

  // display(graph);

  if (containsCycle(graph)) {
    printf("True\n");
  } else {
    printf("False\n");
  }

  return 0;
}

// The createGraph function initializes a new graph with a specified number of
// vertices. It allocates memory for the adjacency list and visited array for
// each vertex setting each list entry to NULL and marking all vertices as
// unvisited
Graph *createGraph(int vertices) {
  // Your code
  Graph *g = (Graph *)malloc(sizeof(Graph));
  g->numVertices = vertices;
  g->adjLists = (Node **)calloc(vertices, sizeof(Node *));
  g->visited = (bool *)calloc(vertices, sizeof(bool));

  for (int i = 0; i < vertices; i++) {
    g->adjLists[i] = (Node *)malloc(sizeof(Node));
    g->adjLists[i]->vertex = i;
    g->adjLists[i]->next = NULL;
  }

  return g;
}

// As the graph is undirected, this function should add an edge from both src to
// dest and dest to src
void addEdge(Graph *graph, int src, int dest) {
  // Your code
  Node *node = (Node *)malloc(sizeof(Node));
  node->vertex = dest;
  node->next = graph->adjLists[src]->next;
  graph->adjLists[src]->next = node;

  node = (Node *)malloc(sizeof(Node));
  node->vertex = src;
  node->next = graph->adjLists[dest]->next;
  graph->adjLists[dest]->next = node;
}

bool bfs(Graph *graph, int startNode, int parentVertex) {
  graph->visited[startNode] = true;
  Node *temp = graph->adjLists[startNode];
  while (temp != NULL) {
    if (!graph->visited[temp->vertex]) {
      bfs(graph, temp->vertex, parentVertex);
    } else {
      if (temp->vertex != parentVertex) {
        return true;
      }
    }
    temp = temp->next;
  }
  return false;
}

// Returns true for cyclic graph, otherwise false
bool containsCycle(Graph *graph) {
  // Your code
  int startNode = 0;

  graph->visited[startNode] = true;
  Node *temp = graph->adjLists[startNode];
  while (temp != NULL) {
    bfs(graph, temp->vertex, temp->vertex);
    temp = temp->next;
  }

  return false;
}
