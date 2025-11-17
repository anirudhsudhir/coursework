#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
  int data;
  struct Node *next;
} Node;

int enqueue(Node **, int);
int dequeue(Node **);
void display(Node *);

int main() {
  Node *tail = NULL;
  enqueue(&tail, 1);
  enqueue(&tail, 2);
  display(tail);
  printf("%d \n", dequeue(&tail));
  return 0;
}

void display(Node *tail) {
  Node *head = tail->next;
  do {
    printf("%d ", head->data);
  } while (head != tail->next);
  printf("\n");
}

int enqueue(Node **tail, int data) {
  Node *new_node = (Node *)malloc(sizeof(Node));
  if (*tail == NULL) {
    return -2;
  }
  if (new_node == NULL) {
    return -1;
  }
  new_node->data = data;
  new_node->next = NULL;
  if ((*tail) == NULL) {
    new_node->next = new_node;
    *tail = new_node;
    return 0;
  }

  new_node->next = (*tail)->next;
  (*tail)->next = new_node;
  (*tail) = new_node;
  return 0;
}

int dequeue(Node **tail) {
  if (*tail == NULL) {
    return -1;
  }
  int data = (*tail)->next->data;
  if ((*tail)->next == (*tail)) {
    free(*tail);
    *tail = NULL;
    return data;
  }

  Node *temp = (*tail)->next;
  (*tail)->next = (*tail)->next->next;
  free(temp);
  return data;
}
