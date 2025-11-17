// Circular Linked List
#include <stdio.h>
#include <stdlib.h>

struct Node {
  int data;
  struct Node *next;
};

struct Node *newNode(int data);
void printList(struct Node *);
void freeList(struct Node *, struct Node *);

struct Node *insertBegin(struct Node *, int);
struct Node *insertPos(struct Node *, int, int);

struct Node *deleteBegin(struct Node *);
struct Node *deletePos(struct Node *, int);

int main() {
  // struct Node *head = newNode(1);
  // head->next = newNode(2);
  // head->next->next = newNode(3);
  // head->next->next->next = newNode(4);
  // struct Node *last = head->next->next->next;
  // last->next = head;

  struct Node *last = insertBegin(NULL, 1);
  last = insertPos(last, 2, 2);
  last = insertPos(last, 3, 3);
  last = insertPos(last, 4, 4);

  printList(last);

  // last = insertBegin(last, 98);
  // printList(last);

  last = insertPos(last, 3, 98);
  printList(last);

  last = deletePos(last, 5);
  printList(last);

  freeList(last, last->next);
  return 0;
}

struct Node *newNode(int data) {
  struct Node *node = (struct Node *)malloc(sizeof(struct Node));
  node->data = data;
  node->next = NULL;

  return node;
}

void printList(struct Node *last) {
  struct Node *head = last->next;

  if (head == NULL) {
    return;
  }

  while (1) {
    printf("%d ", head->data);
    head = head->next;

    if (head == last->next) {
      printf("\n");
      return;
    }
  }
}

void freeList(struct Node *last, struct Node *cur) {
  if (cur->next != last->next) {
    freeList(last, cur->next);
  }

  free(cur);
}

struct Node *insertBegin(struct Node *last, int data) {
  struct Node *node = newNode(data);

  if (last == NULL) {
    node->next = node;
    return node;
  }

  node->next = last->next;
  last->next = node;

  return last;
}

struct Node *insertPos(struct Node *last, int pos, int data) {
  struct Node *node = newNode(data);

  if (last == NULL)
    return insertBegin(NULL, data);

  if (pos == 1) {
    node->next = last->next;
    last->next = node;
    return last;
  }

  struct Node *temp = last->next;
  for (int i = 1; i < pos - 1; i++) {
    if (temp != NULL && temp->next != last->next) {
      temp = temp->next;
    } else {
      return NULL;
    }
  }

  node->next = temp->next;
  temp->next = node;

  // If adding node at the end, return new last node
  if (temp == last)
    return node;

  return last;
}

struct Node *deleteBegin(struct Node *last) {
  if (last == NULL) {
    return NULL;
  }
  if (last == last->next) {
    free(last);
    return NULL;
  }

  struct Node *temp = last->next;
  last->next = last->next->next;

  free(temp);
  return last;
}

struct Node *deletePos(struct Node *last, int pos) {
  if (last == NULL) {
    return NULL;
  }

  struct Node *head = last->next;
  for (int i = 1; i < pos - 1; i++) {
    if (head != NULL && head->next != last->next) {
      head = head->next;
    } else {
      return NULL;
    }
  }

  struct Node *temp = head->next;
  int lastNode = 0;
  if (temp == last) {
    lastNode = 1;
  }
  head->next = head->next->next;
  free(temp);

  if (lastNode) {
    return head;
  }

  return last;
}
