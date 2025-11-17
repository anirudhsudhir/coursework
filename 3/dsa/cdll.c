#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
  int data;
  struct Node *next;
  struct Node *prev;
} Node;

Node *newNode(int);
void printList(Node *);
void printRevList(Node *);
void freeList(Node *, Node *);

Node *insertBegin(Node *, int);
Node *insertPos(Node *, int, int);
Node *insertEnd(Node *, int);

Node *deleteBeg(Node *);
Node *deletePos(Node *, int);
Node *deleteEnd(Node *);

int main() {
  //
  // Node *node1 = newNode(1);
  // Node *node2 = newNode(2);
  // Node *node3 = newNode(3);
  // node1->prev = node3;
  // node1->next = node2;
  // node2->prev = node1;
  // node2->next = node3;
  // node3->prev = node2;
  // node3->next = node1;
  //
  // Node *tail = node3;

  Node *tail = insertBegin(NULL, 3);
  tail = insertBegin(tail, 2);
  tail = insertBegin(tail, 1);
  printList(tail);
  printRevList(tail);

  freeList(tail, tail->next);

  return 0;
}

Node *newNode(int data) {
  Node *node = (Node *)malloc(sizeof(Node));
  node->next = NULL;
  node->prev = NULL;
  node->data = data;

  return node;
}

void printList(Node *tail) {
  if (tail == NULL) {
    return;
  }
  if (tail->next == tail) {
    printf("%d\n", tail->data);
  }

  Node *head = tail->next;
  do {
    printf("%d ", head->data);
    head = head->next;
  } while (tail->next != head);
  printf("\n");
}

void printRevList(Node *tail) {
  if (tail == NULL) {
    return;
  }
  if (tail->next == tail) {
    printf("%d\n", tail->data);
  }

  Node *temp = tail;
  do {
    printf("%d ", temp->data);
    temp = temp->prev;
  } while (temp != tail);
  printf("\n");
}

void freeList(Node *tail, Node *head) {
  if (tail == NULL) {
    return;
  }

  if (head->next != tail->next && head->next != NULL) {
    freeList(tail, head->next);
  }

  free(head);
}

Node *insertBegin(Node *tail, int data) {
  Node *node = newNode(data);

  if (tail == NULL) {
    node->next = node;
    node->prev = node;
  }

  tail->next->prev = node;
  node->next = tail->next;
  tail->next = node;
  node->prev = tail;
  return tail;
}

Node *insertPos(Node *tail, int pos, int data) {
  Node *node = newNode(data);
  if (tail == NULL) {
    node->prev = node;
    node->next = node;
    return node;
  }

  Node *head = tail->next;
  for (int i = 0; i < pos - 1; i++) {
    if (head != NULL) {
      head = head->next;
    }
  }

  node->next = head->next;
  node->prev = head;
  head->next->prev = node;
  head->next = node;

  if (head == tail) {
    return node;
  }

  return tail;
}
