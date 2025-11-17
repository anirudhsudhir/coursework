// Singly Linked List
#include <stdio.h>
#include <stdlib.h>

struct Node {
  int data;
  struct Node *next;
};

void printList(struct Node *);
void freeList(struct Node *);

struct Node *newNode(int);
struct Node *insertBegin(struct Node *, int);
struct Node *insertPosition(struct Node *, int, int);
struct Node *insertEnd(struct Node *, int);

struct Node *deleteBegin(struct Node *);
struct Node *deletePos(struct Node *, int);
struct Node *deleteEnd(struct Node *);
int main() {
  struct Node *head = newNode(1);
  head->next = newNode(2);
  head->next->next = newNode(3);
  head->next->next->next = newNode(4);

  printList(head);

  // head = insertBegin(head, 7);
  // printList(head);

  // head = TestInsertPos(head);
  // printList(head);

  head = insertEnd(head, 98);
  printList(head);

  // head = deletePos(head, 2);
  head = deleteEnd(head);
  printList(head);
  freeList(head);
  return 0;
}

struct Node *newNode(int data) {
  struct Node *node = (struct Node *)malloc(sizeof(struct Node));
  node->data = data;
  node->next = NULL;
  return node;
}

void printList(struct Node *head) {
  while (head != NULL) {
    printf("%d\t", head->data);
    head = head->next;
  }
  printf("\n");
}

struct Node *insertEnd(struct Node *head, int data) {
  struct Node *node = newNode(data);
  struct Node *temp = head;

  while (temp->next != NULL) {
    temp = temp->next;
  }

  temp->next = node;

  return head;
}

void freeList(struct Node *head) {
  if (head->next != NULL)
    freeList(head->next);
  free(head);
}

struct Node *insertBegin(struct Node *head, int data) {
  struct Node *node = newNode(data);
  node->next = head;
  head = node;

  return head;
}

struct Node *insertPosition(struct Node *head, int data, int pos) {
  struct Node *node = newNode(data);
  struct Node *headCopy = head;

  if (pos == 1) {
    node->next = head;
    head = node;
    return head;
  }

  for (int i = 1; i < pos - 1; i++) {
    if (head != NULL) {
      head = head->next;
    } else {
      return NULL;
    }
  }

  node->next = head->next;
  head->next = node;

  return headCopy;
}

struct Node *deleteBegin(struct Node *head) {
  struct Node *temp = head->next;
  free(head);
  return temp;
}

struct Node *deletePos(struct Node *head, int pos) {
  if (pos == 1) {
    return deleteBegin(head);
  }

  struct Node *temp = head;
  for (int i = 1; i < pos - 1; i++) {
    if (temp == NULL) {
      return head;
    } else {
      return head;
    }

    temp = temp->next;
  }

  struct Node *temp2 = temp->next;
  temp->next = temp->next->next;
  free(temp2);

  return head;
}

struct Node *deleteEnd(struct Node *head) {
  struct Node *temp = head;
  if (temp == NULL) {
    return NULL;
  }

  if (temp->next == NULL) {
    free(temp);
    return NULL;
  }

  while (temp->next->next != NULL) {
    temp = temp->next;
  }

  free(temp->next);
  temp->next = NULL;

  return head;
}
